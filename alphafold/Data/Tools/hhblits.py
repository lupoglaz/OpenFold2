from pathlib import Path
from alphafold.Data.Tools import utils
from typing import Optional, Callable, Any, Mapping, Sequence

class HHBlits:
	_DEFAULT_P = 20
	_DEFAULT_Z = 500
	def __init__(self, 
				binary_path: Path,
				databases: Sequence[Path],
				n_cpu: int=4,
				n_iter: int=3,
				e_value: float=1e-3,
				maxseq: int=1e6,
				realign_max: int=1e5,
				maxfilt: int=1e5,
				min_prefilter_hits: int=1000,
				all_seqs: bool=False,
				alt: Optional[int]=None,
				p: int=_DEFAULT_P,
				z: int=_DEFAULT_Z):
		self.binary_path = binary_path
		self.databases = databases
		for database_path in self.databases:
			if not database_path.glob('_*'):
				print(f'HHBlits: Cant find database {database_path}')
				raise ValueError(f'HHBlits: Cant find database {database_path}')
		
		self.n_cpu = n_cpu
		self.n_iter = n_iter
		self.e_value = e_value
		self.maxseq = maxseq
		self.realign_max = realign_max
		self.maxfilt = maxfilt
		self.min_prefilter_hits = min_prefilter_hits
		self.all_seqs = all_seqs
		self.alt = alt
		self.p = p 
		self.z = z 

	def query(self, input_fasta_path: Path) -> Mapping[str, Any]:
		with utils.tmpdir_manager() as query_tmp_dir:
			a3m_path = query_tmp_dir / Path('output.a3m')
			
			db_cmd = []
			for db_path in self.databases:
				db_cmd += ['-d', db_path.as_posix()]
			
			cmd = [
				self.binary_path,
				'-i', input_fasta_path.as_posix(),
				'-cpu', str(self.n_cpu),
				'-oa3m', a3m_path.as_posix(),
				'-o', '/dev/null',
				'-n', str(self.n_iter),
				'-e', str(self.e_value),
				'-maxseq', str(self.maxseq),
				'-realign_max', str(self.realign_max),
				'-maxfilt', str(self.maxfilt),
				'-min_prefilter_hits', str(self.min_prefilter_hits)
			]
			if self.all_seqs:
				cmd += ['-all']
			if self.alt:
				cmd += ['-alt', str(self.alt)]
			if self.p != HHBlits._DEFAULT_P:
				cmd += ['-P', str(self.p)]
			if self.z != HHBlits._DEFAULT_Z:
				cmd += ['-Z', str(self.z)]
			cmd += db_cmd

			print(f'Launching subprocess {"".join(cmd)}')
			process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			with utils.timing(f'HHBlits query'):
				stdout, stderr = process.communicate()
				retcode = process.wait()
			if retcode:
				raise RuntimeError(f"HHBlits failed:\nstdout:\n{stdout.decode('utf-8')}\nstderr:\n{stderr[:500000].decode('utf-8')}")
			
			with open(a3m_path) as f:
				a3m = f.read()

		raw_output = dict(
			a3m=a3m,
			output=stdout,
			stderr=stderr,
			n_iter=self.n_iter,
			e_value=self.e_value
		)
		return raw_output