from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class SmplPipelineContractTest(unittest.TestCase):
    def test_required_output_names_match_assignment(self):
        from src.smpl_lbs_pipeline import REQUIRED_OUTPUTS

        self.assertEqual(
            REQUIRED_OUTPUTS,
            [
                "stage_a_template_weights.png",
                "stage_b_shaped_joints.png",
                "stage_c_pose_offsets.png",
                "stage_d_lbs_result.png",
                "comparison_grid.png",
                "all_joint_weights.png",
                "summary.txt",
            ],
        )

    def test_missing_model_error_names_required_file(self):
        from src.smpl_lbs_pipeline import ModelFileMissing, find_model_file

        self.assertIsNone(find_model_file(["/definitely/not/here"]))
        with self.assertRaisesRegex(ModelFileMissing, "SMPL_NEUTRAL.pkl"):
            raise ModelFileMissing("/tmp/models")


if __name__ == "__main__":
    unittest.main()
