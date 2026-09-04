import unittest

import numpy as np

import xr_ccew as tw


class ProfileTests(unittest.TestCase):
    def test_builtin_reference_profiles_exist(self):
        names = tw.list_wave_profiles()
        self.assertEqual(
            names,
            (
                "KW",
                "n=0 ER",
                "n=1 ER",
                "MRG",
                "MJO",
                "n=0 EIG",
                "n=2 WIG",
                "n=1 WIG",
                "TD-type",
            ),
        )

    def test_alias_and_group_lookup(self):
        kelvin = tw.wave_profile("Kelvin")
        self.assertEqual(kelvin.name, "KW")

        rossby = tw.wave_profiles("Rossby")
        self.assertEqual([profile.name for profile in rossby], ["n=0 ER", "n=1 ER"])

    def test_td_type_profile_matches_kiladis_2006_box(self):
        td = tw.wave_profile("TD-type")
        aliases = ("td", "tdtype", "TD type", "TD-type wave", "tropical depression")
        for alias in aliases + ("easterly wave",):
            self.assertIs(tw.wave_profile(alias), td)

        self.assertEqual(td.wavenumber_bounds, (-20, -6))
        self.assertAlmostEqual(td.frequency_min, 2.0 / 15.0)
        self.assertAlmostEqual(td.frequency_max, 0.5)
        self.assertEqual(td.direction, "westward")
        self.assertEqual(td.symmetry, "both")
        self.assertIsNone(td.curve_name)
        self.assertFalse(np.isfinite(td.equivalent_depth_min))
        self.assertFalse(np.isfinite(td.equivalent_depth_max))
        self.assertEqual(
            td.wavenumber_frequency_polygon,
            ((-20.0, 0.3), (-6.0, 2.0 / 15.0), (-6.0, 1.0 / 3.0), (-20.0, 0.5)),
        )

    def test_polygon_validation(self):
        kelvin = tw.wave_profile("KW")
        with self.assertRaises(ValueError):
            kelvin.with_updates(wavenumber_frequency_polygon=((1, 0.1), (2, 0.2)))
        with self.assertRaises(ValueError):  # bow-tie is not convex
            kelvin.with_updates(
                wavenumber_frequency_polygon=((1, 0.1), (5, 0.3), (5, 0.1), (1, 0.3))
            )
        with self.assertRaises(ValueError):
            kelvin.with_updates(wavenumber_frequency_polygon=((1, 0.1), (5, np.nan), (5, 0.3)))

        custom = kelvin.with_updates(wavenumber_frequency_polygon=[(1, 0.1), (5, 0.1), (5, 0.3)])
        self.assertEqual(custom.wavenumber_frequency_polygon, ((1.0, 0.1), (5.0, 0.1), (5.0, 0.3)))
        self.assertIsNone(kelvin.wavenumber_frequency_polygon)

    def test_profile_copy_updates(self):
        kelvin = tw.wave_profile("KW")
        custom = kelvin.with_updates(frequency_max=0.25)
        self.assertEqual(kelvin.frequency_max, 0.4)
        self.assertEqual(custom.frequency_max, 0.25)


if __name__ == "__main__":
    unittest.main()

