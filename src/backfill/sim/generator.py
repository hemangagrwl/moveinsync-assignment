from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from backfill.core.geo import METRES_PER_DEGREE, haversine_np
from backfill.core.models import (
    GeoPoint,
    PingBatch,
    Rider,
    RiderStatus,
    RiderType,
    TripEvidence,
    TripType,
)

BASE_TIME_MS = 1_767_225_600_000  # 2026-01-01T00:00:00Z


@dataclass(frozen=True)
class GenConfig:
    target_pings: int = 1500
    noise_m: float = 3.0
    low_accuracy_rate: float = 0.02
    duplicate_rate: float = 0.01
    teleport_prob: float = 0.3
    gaps: tuple[int, int] = (1, 3)
    gap_minutes: tuple[float, float] = (3.0, 20.0)
    gap_over_rider_prob: float = 0.6
    basement_prob: float = 0.5
    riders: tuple[int, int] = (2, 5)
    actual_valid_rate: float = 0.6
    adhoc_rate: float = 0.2
    planned_time_sigma_s: float = 180.0
    speed_kmh: tuple[float, float] = (18.0, 38.0)


@dataclass(frozen=True)
class SyntheticTrip:
    evidence: TripEvidence
    true_distance_m: float
    naive_distance_m: float
    gaps_s: tuple[tuple[float, float], ...] = field(default=())


class City:
    """A Bengaluru-sized box with fixed offices and a fixed pool of employee homes.

    Employees and offices repeat across trips, exactly like real rosters, which is what gives
    the routing cache something to hit.
    """

    LAT = (12.90, 13.05)
    LON = (77.52, 77.72)

    def __init__(self, seed: int = 7, offices: int = 5, employees: int = 2000) -> None:
        rng = np.random.default_rng(seed)
        self.offices = [self._point(rng) for _ in range(offices)]
        self.homes = [self._point(rng) for _ in range(employees)]

    def _point(self, rng: np.random.Generator) -> GeoPoint:
        return GeoPoint(float(rng.uniform(*self.LAT)), float(rng.uniform(*self.LON)))


class TripGenerator:
    """Drives a cab along a street grid and records what a flaky GPS device would report."""

    def __init__(self, city: City | None = None, config: GenConfig | None = None, seed: int = 11) -> None:
        self.city = city or City()
        self.cfg = config or GenConfig()
        self.seed = seed

    def generate(self, index: int) -> SyntheticTrip:
        rng = np.random.default_rng((self.seed, index))
        cfg = self.cfg
        trip_type = TripType.LOGOUT if rng.random() < 0.6 else TripType.LOGIN
        office = self.city.offices[int(rng.integers(len(self.city.offices)))]
        n_riders = int(rng.integers(cfg.riders[0], cfg.riders[1] + 1))
        homes_idx = rng.choice(len(self.city.homes), size=n_riders, replace=False)
        refs = {self.city.homes[int(i)]: f"E{int(i):05d}" for i in homes_idx}
        homes = list(refs)

        if trip_type is TripType.LOGOUT:
            visited = self._nearest_order(office, homes)
            stops = [office, *visited]
        else:
            visited = self._nearest_order(office, homes)[::-1]
            first = visited[0]
            start = GeoPoint(first.lat + rng.uniform(-0.01, 0.01), first.lon + rng.uniform(-0.01, 0.01))
            stops = [start, *visited, office]

        v_lat, v_lon, leg_end_vertex = self._staircase(rng, stops)
        v_cum = np.concatenate([[0.0], np.cumsum(haversine_np(v_lat[:-1], v_lon[:-1], v_lat[1:], v_lon[1:]))])
        true_m = float(v_cum[-1])

        key_t, key_d, arrivals = self._timeline(rng, v_cum, leg_end_vertex)
        duration_s = float(key_t[-1])

        # Real devices report every few seconds; short trips simply get fewer pings.
        interval = min(max(duration_s / cfg.target_pings, 2.0), 6.0)
        n = round(duration_s / interval)
        t = np.linspace(0.0, duration_s, n) + rng.uniform(-0.4, 0.4, n) * interval
        t = np.clip(t, 0.0, duration_s)
        dist = np.interp(t, key_t, key_d)
        lat = np.interp(dist, v_cum, v_lat)
        lon = np.interp(dist, v_cum, v_lon)

        # GPS error drifts slowly rather than jumping independently each second, so the noise is
        # white noise smoothed over ~20 s, rescaled back to noise_m.
        window = max(1, round(20.0 / interval))
        kernel = np.ones(window) / math.sqrt(window)
        e_lat = np.convolve(rng.normal(0.0, 1.0, n + window), kernel, "same")[:n] * cfg.noise_m
        e_lon = np.convolve(rng.normal(0.0, 1.0, n + window), kernel, "same")[:n] * cfg.noise_m
        acc = rng.uniform(3.0, 15.0, n)
        low = rng.random(n) < cfg.low_accuracy_rate
        e_lat[low] = rng.normal(0.0, 50.0, int(low.sum()))
        e_lon[low] = rng.normal(0.0, 50.0, int(low.sum()))
        acc[low] = rng.uniform(60.0, 200.0, int(low.sum()))
        lat = lat + e_lat / METRES_PER_DEGREE
        lon = lon + e_lon / (METRES_PER_DEGREE * math.cos(math.radians(office.lat)))

        gaps = self._gaps(rng, trip_type, duration_s, arrivals)
        in_gap = np.zeros(n, dtype=bool)
        for g0, g1 in gaps:
            in_gap |= (t >= g0) & (t <= g1)
        keep = ~in_gap
        t, lat, lon, acc = t[keep], lat[keep], lon[keep], acc[keep]

        if rng.random() < cfg.teleport_prob and len(t) > 10:
            t, lat, lon, acc = self._add_teleport(rng, t, lat, lon, acc)

        ts_ms = BASE_TIME_MS + index * 86_400_000 + np.round(t * 1000).astype(np.int64)
        dup = np.flatnonzero(rng.random(len(ts_ms)) < cfg.duplicate_rate)
        order = rng.permutation(len(ts_ms) + len(dup))
        all_ts = np.concatenate([ts_ms, ts_ms[dup]])[order]
        all_lat = np.concatenate([lat, lat[dup]])[order]
        all_lon = np.concatenate([lon, lon[dup]])[order]
        all_acc = np.concatenate([acc, acc[dup]])[order]
        pings = PingBatch.from_arrays(all_ts, all_lat, all_lon, np.round(all_acc, 1))

        start_ms = BASE_TIME_MS + index * 86_400_000
        end_ms = start_ms + round(duration_s * 1000)
        rider_arrivals = arrivals[:-1] if trip_type is TripType.LOGIN else arrivals
        riders = self._riders(rng, visited, refs, rider_arrivals, start_ms)
        start_loc, end_loc = self._edge_locations(rng, trip_type, office, stops)

        evidence = TripEvidence(
            trip_id=f"T{index:06d}",
            trip_type=trip_type,
            office=office,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            pings=pings,
            riders=riders,
            start_location=start_loc,
            end_location=end_loc,
        )
        return SyntheticTrip(evidence, true_m, naive_distance_m(pings), tuple(gaps))

    # -- path ---------------------------------------------------------------------------

    @staticmethod
    def _nearest_order(origin: GeoPoint, points: list[GeoPoint]) -> list[GeoPoint]:
        remaining = list(points)
        out: list[GeoPoint] = []
        cur = origin
        while remaining:
            nxt = min(remaining, key=lambda p: abs(p.lat - cur.lat) + abs(p.lon - cur.lon))
            remaining.remove(nxt)
            out.append(nxt)
            cur = nxt
        return out

    @staticmethod
    def _staircase(
        rng: np.random.Generator, stops: list[GeoPoint]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int]]:
        """Monotone axis-aligned paths between stops: the shortest route on a street grid."""
        lats, lons = [stops[0].lat], [stops[0].lon]
        leg_end: list[int] = []
        for a, b in itertools.pairwise(stops):
            turns = int(rng.integers(1, 4))
            f_lat = rng.dirichlet(np.ones(turns))
            f_lon = rng.dirichlet(np.ones(turns))
            lat_first = rng.random() < 0.5
            cur_lat, cur_lon = a.lat, a.lon
            for k in range(turns):
                moves = ("lat", "lon") if lat_first else ("lon", "lat")
                for m in moves:
                    if m == "lat":
                        cur_lat += (b.lat - a.lat) * f_lat[k]
                    else:
                        cur_lon += (b.lon - a.lon) * f_lon[k]
                    lats.append(cur_lat)
                    lons.append(cur_lon)
            lats[-1], lons[-1] = b.lat, b.lon
            leg_end.append(len(lats) - 1)
        return np.array(lats), np.array(lons), leg_end

    def _timeline(
        self, rng: np.random.Generator, v_cum: NDArray[np.float64], leg_end: list[int]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], list[float]]:
        """Piecewise-constant speed with traffic stops and dwell at every stop."""
        t, d = float(rng.uniform(30, 90)), 0.0
        kt, kd = [0.0, t], [0.0, 0.0]
        arrivals: list[float] = []
        for j, vi in enumerate(leg_end):
            leg_len = float(v_cum[vi]) - d
            v = rng.uniform(*self.cfg.speed_kmh) / 3.6
            for f in np.sort(rng.uniform(0.1, 0.9, int(rng.integers(0, 3)))):
                target = d + leg_len * float(f)
                t += (target - kd[-1]) / v
                kt.append(t)
                kd.append(target)
                t += float(rng.uniform(20, 90))
                kt.append(t)
                kd.append(target)
            end = float(v_cum[vi])
            t += (end - kd[-1]) / v
            kt.append(t)
            kd.append(end)
            arrivals.append(t)
            dwell = float(rng.uniform(60, 150)) if j < len(leg_end) - 1 else float(rng.uniform(20, 60))
            t += dwell
            kt.append(t)
            kd.append(end)
            d = end
        return np.array(kt), np.array(kd), arrivals

    # -- device behaviour ------------------------------------------------------------------

    def _gaps(
        self, rng: np.random.Generator, trip_type: TripType, duration: float, arrivals: list[float]
    ) -> list[tuple[float, float]]:
        cfg = self.cfg
        gaps: list[tuple[float, float]] = []
        if trip_type is TripType.LOGOUT and rng.random() < cfg.basement_prob:
            gaps.append((0.0, float(rng.uniform(120, 300))))
        rider_arrivals = arrivals[:-1] if trip_type is TripType.LOGIN else arrivals
        for _ in range(int(rng.integers(cfg.gaps[0], cfg.gaps[1] + 1))):
            length = float(rng.uniform(*cfg.gap_minutes)) * 60
            if rider_arrivals and rng.random() < cfg.gap_over_rider_prob:
                centre = rider_arrivals[int(rng.integers(len(rider_arrivals)))]
                g0 = centre - length * float(rng.uniform(0.2, 0.8))
            else:
                g0 = float(rng.uniform(0, max(1.0, duration - length)))
            gaps.append((max(0.0, g0), min(duration, g0 + length)))
        return gaps

    @staticmethod
    def _add_teleport(
        rng: np.random.Generator,
        t: NDArray[np.float64],
        lat: NDArray[np.float64],
        lon: NDArray[np.float64],
        acc: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        i = int(rng.integers(5, len(t) - 5))
        k = int(rng.integers(3, 6))
        bt = t[i] + 0.5 + np.arange(k) * 1.0
        jump = rng.uniform(5_000, 10_000) / METRES_PER_DEGREE
        blat = np.full(k, lat[i] + jump)
        blon = np.full(k, lon[i] + jump)
        bacc = np.full(k, 8.0)
        return (
            np.concatenate([t, bt]),
            np.concatenate([lat, blat]),
            np.concatenate([lon, blon]),
            np.concatenate([acc, bacc]),
        )

    def _riders(
        self,
        rng: np.random.Generator,
        visited: list[GeoPoint],
        refs: dict[GeoPoint, str],
        arrivals: list[float],
        start_ms: int,
    ) -> tuple[Rider, ...]:
        cfg = self.cfg
        out: list[Rider] = []
        for home, arrival in zip(visited, arrivals, strict=True):
            event_ms = start_ms + round(arrival * 1000)
            planned_ms = event_ms + round(float(rng.normal(0, cfg.planned_time_sigma_s)) * 1000)
            adhoc = rng.random() < cfg.adhoc_rate
            roll = rng.random()
            actual: GeoPoint | None
            if roll < cfg.actual_valid_rate:
                actual = GeoPoint(
                    home.lat + float(rng.normal(0, 15)) / METRES_PER_DEGREE,
                    home.lon + float(rng.normal(0, 15)) / METRES_PER_DEGREE,
                )
            elif roll < cfg.actual_valid_rate + (1 - cfg.actual_valid_rate) / 3:
                actual = None
            elif roll < cfg.actual_valid_rate + 2 * (1 - cfg.actual_valid_rate) / 3:
                actual = GeoPoint(0.0, 0.0)
            else:
                actual = GeoPoint(home.lat + 0.05, home.lon + 0.05)
            out.append(
                Rider(
                    employee_ref=refs[home],
                    rider_type=RiderType.ADHOC if adhoc else RiderType.PLANNED,
                    status=RiderStatus.TRAVELLED,
                    planned_location=None if adhoc else home,
                    planned_time_ms=None if adhoc else planned_ms,
                    actual_location=actual,
                    actual_time_ms=event_ms,
                    profile_home=home if adhoc else None,
                    sign_in_time_ms=planned_ms if adhoc else None,
                )
            )
        return tuple(out)

    def _edge_locations(
        self, rng: np.random.Generator, trip_type: TripType, office: GeoPoint, stops: list[GeoPoint]
    ) -> tuple[GeoPoint | None, GeoPoint | None]:
        def near(p: GeoPoint, metres: float) -> GeoPoint:
            return GeoPoint(p.lat + metres / METRES_PER_DEGREE, p.lon)

        office_actual = (
            near(office, float(rng.uniform(0, 30))) if rng.random() < 0.85 else near(office, 900.0)
        )
        if trip_type is TripType.LOGOUT:
            return office_actual, near(stops[-1], float(rng.uniform(0, 30)))
        return near(stops[0], float(rng.uniform(0, 30))), office_actual


def naive_distance_m(pings: PingBatch) -> float:
    """What a naive system bills: sum of straight lines between raw pings in time order."""
    if len(pings) < 2:
        return 0.0
    order = np.argsort(pings.ts_ms, kind="stable")
    lat, lon = pings.lat[order], pings.lon[order]
    return float(haversine_np(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum())
