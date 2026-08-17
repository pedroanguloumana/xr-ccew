import unittest

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
            ),
        )

    def test_alias_and_group_lookup(self):
        kelvin = tw.wave_profile("Kelvin")
        self.assertEqual(kelvin.name, "KW")

        rossby = tw.wave_profiles("Rossby")
        self.assertEqual([profile.name for profile in rossby], ["n=0 ER", "n=1 ER"])

    def test_profile_copy_updates(self):
        kelvin = tw.wave_profile("KW")
        custom = kelvin.with_updates(frequency_max=0.25)
        self.assertEqual(kelvin.frequency_max, 0.4)
        self.assertEqual(custom.frequency_max, 0.25)


if __name__ == "__main__":
    unittest.main()

