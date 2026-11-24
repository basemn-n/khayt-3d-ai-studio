import unittest
from main import validate_design

class DesignValidationTests(unittest.TestCase):
    def valid(self):
        return {"summary":"test", "objects":[{"name":"box","type":"tatami","closed":True,"color":"#176447","fill":"tatami","spacing":0.45,"stitch_length":3.2,"max_stitch_length":7,"angle":0,"satin_width":3,"points":[{"x":0.1,"y":0.1,"curve":False},{"x":0.9,"y":0.1,"curve":False},{"x":0.9,"y":0.9,"curve":False},{"x":0.1,"y":0.9,"curve":False}]}]}
    def test_valid_design(self):
        self.assertEqual(len(validate_design(self.valid())["objects"]), 1)
    def test_reject_outside_safe_margin(self):
        d=self.valid(); d["objects"][0]["points"][0]["x"]=0.01
        with self.assertRaises(ValueError): validate_design(d)
    def test_reject_duplicate_points(self):
        d=self.valid(); d["objects"][0]["points"][1]["x"]=0.1001; d["objects"][0]["points"][1]["y"]=0.1001
        with self.assertRaises(ValueError): validate_design(d)
    def test_clamps_stitch_parameters(self):
        d=self.valid(); d["objects"][0]["spacing"]=99
        self.assertEqual(validate_design(d)["objects"][0]["spacing"],1.2)
    def test_reject_self_intersecting_polygon(self):
        d=self.valid(); d["objects"][0]["points"]=[{"x":0.1,"y":0.1,"curve":False},{"x":0.9,"y":0.9,"curve":False},{"x":0.1,"y":0.9,"curve":False},{"x":0.9,"y":0.1,"curve":False}]
        with self.assertRaises(ValueError): validate_design(d)

if __name__ == "__main__": unittest.main()
