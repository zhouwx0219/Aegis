import pathlib
import tomllib
import unittest


class ProjectMetadataTests(unittest.TestCase):
    def test_pyproject_declares_runtime_dependency_and_cli(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        with (root / "pyproject.toml").open("rb") as handle:
            data = tomllib.load(handle)

        self.assertEqual(data["project"]["name"], "astra-core")
        self.assertIn("pybind11>=2.10", data["project"]["dependencies"])
        self.assertEqual(
            data["project"]["scripts"]["astra-cc-matrix"],
            "agent.evaluation.cc_matrix_cli:main",
        )
        self.assertEqual(
            data["project"]["scripts"]["astra-dbx1000-native"],
            "agent.evaluation.dbx1000_native:main",
        )
        self.assertEqual(
            data["project"]["scripts"]["astra-agent-paths"],
            "agent.evaluation.agent_path_experiment:main",
        )
        self.assertEqual(
            data["project"]["scripts"]["astra-atcc-retry"],
            "agent.evaluation.atcc_retry_experiment:main",
        )
        self.assertEqual(
            data["project"]["scripts"]["astra-atcc-train"],
            "agent.evaluation.atcc_policy_training:main",
        )
        self.assertEqual(
            data["project"]["scripts"]["astra-atcc-profiles"],
            "agent.evaluation.atcc_profile_runner:main",
        )
        self.assertEqual(
            data["project"]["scripts"]["astra-atcc-cost-search"],
            "agent.evaluation.atcc_reward_cost_search:main",
        )
        self.assertEqual(
            data["project"]["scripts"]["astra-atcc-ablation"],
            "agent.evaluation.atcc_ablation_experiment:main",
        )
        self.assertEqual(
            data["tool"]["setuptools"]["packages"]["find"]["include"],
            ["agent*"],
        )


if __name__ == "__main__":
    unittest.main()
