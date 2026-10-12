"""Speed-adaptive foveation: grow the finest zone with the distance the vehicle needs to react and stop.

near radius = clamp(v * t_react + v^2 / (2 * decel) + margin, base radius, max_near), on the snap grid.
It grows immediately (safety first) and shrinks only after the target has stayed smaller for
`shrink_delay` seconds (hysteresis, so the map isn't resampled on every speed wobble).
Only the first zone changes; outer zones keep their radii (the first must stay inside the second).
"""
import math


class ZonePolicy:
    def __init__(self, zones, cfg, snap):
        self.base = [dict(z) for z in zones]
        self.cfg, self.snap = cfg, snap
        self.current = zones[0]["half_extent"]
        self.limit = min(cfg.get("max_near", 25.0), zones[1]["half_extent"] - snap) if len(zones) > 1 else cfg.get("max_near", 25.0)
        self._shrink_since = None

    def target(self, speed):
        c = self.cfg
        v = abs(speed)
        d = v * c.get("reaction_time", 1.0) + v * v / (2 * c.get("decel", 2.0)) + c.get("margin", 4.0)
        d = math.ceil(d / self.snap) * self.snap
        return min(max(d, self.base[0]["half_extent"]), self.limit)

    def update(self, speed, t):
        """Returns a new zone list when the near radius should change, else None."""
        want = self.target(speed)
        if want > self.current:
            self._shrink_since = None
            return self._set(want)
        if want < self.current:
            if self._shrink_since is None:
                self._shrink_since = t
            elif t - self._shrink_since >= self.cfg.get("shrink_delay", 3.0):
                self._shrink_since = None
                return self._set(want)
        else:
            self._shrink_since = None
        return None

    def _set(self, r):
        self.current = r
        zones = [dict(z) for z in self.base]
        zones[0]["half_extent"] = r
        return zones
