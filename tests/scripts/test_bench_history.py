import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import bench_history


class HistoryOrderingTests(unittest.TestCase):
    def test_results_follow_numeric_run_order(self):
        with tempfile.TemporaryDirectory() as directory:
            for index in (0, 1, 2, 9, 10, 11):
                (Path(directory) / f"run_{index}.json").touch()
            (Path(directory) / "run_other.json").touch()

            expected = [f"run_{i}.json" for i in (0, 1, 2, 9, 10, 11)] + ["run_other.json"]
            for finder in (bench_history._find_existing_results, bench_history._find_result_files):
                self.assertEqual(expected, [Path(path).name for path in finder("run", directory)])


if __name__ == "__main__":
    unittest.main()
