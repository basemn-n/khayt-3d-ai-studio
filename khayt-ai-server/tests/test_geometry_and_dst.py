from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


def circle(cx: float, cy: float, rx: float, ry: float, count: int = 12):
    import math
    return [
        {
            "x": cx + rx * math.cos(2 * math.pi * i / count),
            "y": cy + ry * math.sin(2 * math.pi * i / count),
            "curve": True,
        }
        for i in range(count)
    ]


class GeometryValidationTests(unittest.TestCase):
    def test_valid_mixed_embroidery_geometry(self):
        payload = {
            "summary": "اختبار وردة",
            "objects": [
                {"name": "بتلة", "type": "tatami", "points": circle(.5, .3, .12, .16), "closed": True, "color": "#d92d45"},
                {"name": "ساق", "type": "run", "points": [
                    {"x": .5, "y": .45}, {"x": .51, "y": .6},
                    {"x": .49, "y": .75}, {"x": .5, "y": .9},
                ], "closed": False, "color": "#23834b"},
                {"name": "عمود ساتان", "type": "satin", "points": [
                    {"x": .3, "y": .5}, {"x": .35, "y": .55},
                    {"x": .4, "y": .6}, {"x": .45, "y": .65},
                ], "closed": False, "color": "#23834b"},
            ],
        }
        result = main.normalize_design(payload, 120, 150)
        self.assertEqual(len(result["objects"]), 3)
        self.assertEqual(result["objects"][0]["fill"], "tatami")
        self.assertEqual(result["objects"][1]["fill"], "run")
        self.assertEqual(result["objects"][2]["fill"], "satin")

    def test_rejects_unsupported_complex_type(self):
        payload = {"objects": [{
            "name": "شكل", "type": "complex", "points": circle(.5, .5, .2, .2),
            "closed": True, "color": "#123456",
        }]}
        with self.assertRaises(main.HTTPException):
            main.normalize_design(payload, 100, 100)

    def test_rejects_satin_as_closed_polygon(self):
        payload = {"objects": [{
            "name": "ساتان", "type": "satin", "points": circle(.5, .5, .2, .2),
            "closed": True, "color": "#123456",
        }]}
        with self.assertRaises(main.HTTPException):
            main.normalize_design(payload, 100, 100)

    def test_rejects_degenerate_closed_shape(self):
        points = [{"x": .5, "y": .5 + i * .001, "curve": False} for i in range(9)]
        payload = {"objects": [{
            "name": "خط", "type": "tatami", "points": points,
            "closed": True, "color": "#123456",
        }]}
        with self.assertRaises(main.HTTPException):
            main.normalize_design(payload, 100, 100)


class DSTExportTests(unittest.TestCase):
    def test_exports_valid_dst_header_and_records(self):
        request = main.DSTExportRequest(
            name="Khayt Test",
            width_mm=100,
            height_mm=100,
            initial_color="#176447",
            stitches=[
                {"x": 1, "y": 1, "jump": True},
                {"x": 10, "y": 1},
                {"x": 10, "y": 10},
                {"x": 1, "y": 10},
            ],
        )
        response = asyncio.run(main.export_dst(request))
        self.assertEqual(response.media_type, "application/octet-stream")
        self.assertGreaterEqual(len(response.body), 515)
        self.assertEqual(len(response.body) % 3, 2)
        self.assertTrue(response.body.startswith(b"LA:"))

    def test_rejects_stitch_outside_fabric(self):
        request = main.DSTExportRequest(
            name="Out of bounds",
            width_mm=20,
            height_mm=20,
            stitches=[{"x": 1, "y": 1}, {"x": 30, "y": 1}],
        )
        with self.assertRaises(main.HTTPException):
            asyncio.run(main.export_dst(request))


if __name__ == "__main__":
    unittest.main()
