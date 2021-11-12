from pathlib import Path
from alphafold.Data.Tools import utils
from typing import Sequence

class HHSearch:
	def __init__(self,
				binary_path: Path,
				databases: Sequence[Path],
				maxseq: int=1e6):
		self.binary_path = binary_path,
		self.databases = databases
		self.maxseq = maxseq

		for database_path in self.databases:
			if not database_path.glob('_*'):
				print(f'HHSearch: Cant find database {database_path}')
				raise ValueError(f'HHSearch: Cant find database {database_path}')

	def query(self, a3m: str) -> str:
		with utils.tmpdir_manager() as query_tmp_dir:
			input_path = query_tmp_dir / Path('query.a3m')
			hhr_path = query_tmp_dir / Path('output.hhr')
			with open(input_path, 'w') as f:
				f.write(a3m)

			db_cmd = []
			for db_path in self.databases:
				db_cmd += ['-d', db_path.as_posix()]
			cmd = [
				self.binary_path.as_posix(),
				'-i', input_path.as_posix(),
				'-o', hhr_path.as_posix(),
				'-maxseq', str(self.maxseq)
			] + db_cmd

			print(f'Launching subprocess {''.join(cmd)}')
			process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			with utils.timing(f'HHSearch query'):
				stdout, stderr = process.communicate()
				retcode = process.wait()
			if retcode:
				raise RuntimeError(f'HHSearch failed:\nstdout:\n{stdout.decode('utf-8')}\nstderr:\n{stderr[:100000].decode('utf-8')}')

			with open(hhr_path) as f:
				hhr = f.read()
				
		return hhr
