"""No-GPU regression checks for source restoration, recipe routing and result QA."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class ReleaseTests(unittest.TestCase):
    def test_three_frozen_runtimes(self):
        m=module('scripts/reproduce/materialize_einv2_release.py','restore')
        versions=list((ROOT/'provenance/runtime_snapshots').iterdir())
        self.assertEqual(len(versions),3)
        for v in versions:
            with tempfile.TemporaryDirectory() as tmp:
                output=Path(tmp)/'runtime'
                self.assertGreater(m.materialize(v.name,output),40)
                for rel,content in m.resolve_files(v.name).items():
                    self.assertEqual((output/rel).read_bytes(),content)
                with self.assertRaises(FileExistsError):m.materialize(v.name,output)

    def test_current_runtime_matches_dynamicmask(self):
        m=module('scripts/reproduce/materialize_einv2_release.py','restore_current')
        for rel,content in m.resolve_files('einv2_rb05_dynamicmask_v1').items():
            self.assertEqual((ROOT/rel).read_bytes(),content,rel)

    def test_named_recipes_dry_run(self):
        m=module('scripts/train/einv2_recipe.py','recipes')
        paths=json.loads((ROOT/'configs/hosts/RB05.example.json').read_text())
        self.assertIn('/shared/_hdf5',paths['hdf5_dir'])
        recipes=list((ROOT/'configs/einv2/audited').glob('*.json'))
        self.assertEqual(len(recipes),8)
        for p in recipes:
            r=json.loads(p.read_text());cmd=m.build_command(r,paths,2026,'/not-created')
            self.assertIn('einv2_audited.py',cmd[1]);self.assertNotIn('--split',cmd)
            result=subprocess.run([sys.executable,str(ROOT/'scripts/train/einv2_recipe.py'),
                                   '--recipe',str(p),'--paths',str(ROOT/'configs/hosts/RB05.example.json'),
                                   '--seed','2026','--output-root','/not-created','--dry-run'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_result_snapshot_qa(self):
        m=module('scripts/reports/build_all_results.py','results')
        m.main()
        qa=json.loads((ROOT/'reports/summary_20260907/qa.json').read_text())
        self.assertEqual(qa['formal_main_runs'],68)
        self.assertEqual(qa['scored_run_splits'],131)
        self.assertEqual(len(qa['missing_evaluation']),5)
        self.assertEqual(qa['failed_smokes'],2)


if __name__=='__main__':unittest.main(verbosity=2)
