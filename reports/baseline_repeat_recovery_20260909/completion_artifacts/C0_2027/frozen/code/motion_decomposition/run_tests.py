import json
from pathlib import Path
import sys
import time
import unittest

start=time.time();suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern='test_*.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
with Path(sys.argv[1]).open('x') as f:json.dump(dict(testsRun=result.testsRun,failures=[str(x) for x in result.failures],errors=[str(x) for x in result.errors],successful=result.wasSuccessful(),elapsed=time.time()-start),f,indent=2)
sys.exit(not result.wasSuccessful())
