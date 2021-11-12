from pathlib import Path
from alphafold.Data.Tools import utils
from typing import Optional

class HMMSearch:
	def __init__(self,
				binary_path: Path,
				database_path: Path,
				n_cpu: int=8,
				flags: Optional[Sequence[str]]=None):
		self.binary_path = binary_path
		self.database_path = database_path
		self.n_cpu = n_cpu
		self.flags = flags
		
		if (not database_path.exists()):
			print(f'HMMSearch database {database_path} not found')
			raise ValueError(f'HMMSearch database {database_path} not found')
	
	def query(self, hmm: str) -> str:
		with utils.tmpdir_manager() as query_tmp_dir:
			hmm_input_path = query_tmp_dir / Path('query.hmm')
			output_a3m_path = query_tmp_dir / Path('output.a3m')
			with open(hmm_input_path, 'w') as f:
				f.write(hmm)
			
			cmd = [
				self.binary_path.as_posix(),
				'--noali',
				'--cpu', str(self.n_cpu)
			]

			if self.flags:
				cmd += self.flags

			cmd += [
				'-A', output_a3m_path.as_posix(),
				hmm_input_path.as_posix(),
				self.database_path
			]

			print(f'Launching subprocess {''.join(cmd)}')
			process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			with utils.timing(f'HMMSearch query'):
				stdout, stderr = process.communicate()
				retcode = process.wait()
			if retcode:
				raise RuntimeError(f'HMMSearch failed:\nstdout:\n{stdout.decode('utf-8')}\nstderr:\n{stderr.decode('utf-8')}')
			
			with open(output_a3m_path) as f:
				a3m = f.read()
		
		return a3m