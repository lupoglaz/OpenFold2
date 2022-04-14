from pathlib import Path
from alphafold.Data.Tools import utils
from alphafold.Data import parsers
from typing import Optional, Callable, Any, Mapping, Sequence
import subprocess

class Jackhammer:
	def __init__(self, 
				binary_path:Path, database_path:Path,
				n_cpu:int=8, n_iter:int=1,
				e_value:float=1e-4, z_value:Optional[float]=None, filter_f1:float=5e-4,filter_f2:float=5e-5,filter_f3:float=5e-6,
				get_tblout:bool=False, incdom_e:Optional[float]=None, dom_e:Optional[float]=None,
				num_streamed_chunks:Optional[int]=None,
				streaming_callback:Optional[Callable[[int], None]]=None):
		self.binary_path = binary_path
		self.database_path = database_path
		if (not database_path.exists()):
			print(f'Jackhammer database {database_path} not found')
			raise ValueError(f'Jackhammer database {database_path} not found')

		self.n_cpu = n_cpu
		self.n_iter = n_iter
		self.e_value = e_value
		self.z_value = z_value
		self.filter_f1 = filter_f1
		self.filter_f2 = filter_f2
		self.filter_f3 = filter_f3
		self.get_tblout = get_tblout
		self.dom_e = dom_e
		self.incdom_e = incdom_e
		self.num_streamed_chunks = num_streamed_chunks
		self.streaming_callback = streaming_callback

	def _query_chunk(self, input_fasta_path:Path, database_path:Path, max_sequences:Optional[int]=None) -> Mapping[str, Any]:
		with utils.tmpdir_manager() as query_tmp_dir:
			sto_path = query_tmp_dir / Path('output.sto')
			cmd_flags = [
				'-o', '/dev/null',
				'-A', sto_path.as_posix(),
				'--noali',
				'--F1', str(self.filter_f1),
				'--F2', str(self.filter_f2),
				'--F3', str(self.filter_f3),
				'--incE', str(self.e_value),
				'-E', str(self.e_value),
				'--cpu', str(self.n_cpu),
				'-N', str(self.n_iter)
			]

			if self.get_tblout:
				tblout_path = query_tmp_dir / Path('tblout.txt')
				cmd_flags.extend(['-tblout', tblout_path.as_posix()])

			if not (self.z_value is None):
				cmd_flags.extend(['-Z', str(self.z_value)])

			if not (self.dom_e is None):
				cmd_flags.extend(['--domE', str(self.dom_e)])
			
			if not (self.incdom_e is None):
				cmd_flags.extend(['--incdomE', str(self.incdom_e)])

			cmd = [self.binary_path.as_posix()] + cmd_flags + [input_fasta_path.as_posix(), database_path.as_posix()]
			print(f"Launching subprocess {''.join(cmd)}")
			process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			with utils.timing(f'Jackhammer {database_path.name} query'):
				_, stderr = process.communicate()
				retcode = process.wait()
			if retcode:
				raise RuntimeError(f"Jackhammer failed: {stderr.decode('utf-8')}")
			
			tbl=''
			if self.get_tblout:
				with open(tblout_path) as f:
					tbl = f.read()
			
			if max_sequences is None:
				with open(sto_path) as f:
					sto = f.read()
			else:
				sto = parsers.truncate_stockholm_msa(sto_path, max_sequences)

			raw_output = dict(
				sto = sto,
				tbl = tbl,
				stderr = stderr,
				n_iter = self.n_iter,
				e_value = self.e_value
			)
			return raw_output

	def query(self, input_fasta_path:Path, max_sequences:Optional[int]=None) -> Sequence[Mapping[str, Any]]:
		if self.num_streamed_chunks is None:
			return [self._query_chunk(input_fasta_path, self.database_path, max_sequences=max_sequences)]
		
		raise NotImplemented()

if __name__ == '__main__':
	pass


