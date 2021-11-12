import contextlib
import shutil
import tempfile
import time
from typing import Optional
from pathlib import Path

@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str]=None):
	tmpdir = tempfile.mkdtemp(dir=base_dir)
	try:
		yield Path(tmpdir)
	finally:
		shutil.rmtree(tmpdir, ignore_errors=True)

@contextlib.contextmanager
def timing(msg: str):
	print(f'Started {msg}')
	tic = time.time()
	yield
	tac = time.time()
	print(f'Finished {msg} in {tac-tic}')

if __name__=='__main__':
	with tmpdir_manager() as f:
		print(f)

	with timing("Timing test"):
		time.sleep(1.0)
	


