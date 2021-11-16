from pathlib import Path
from alphafold.Data.Tools import utils
from typing import Sequence

class Kalign:
	def __init__(self,
				binary_path: Path):
		self.binary_path = binary_path

	def align(self, sequences: Sequence[str]) -> str:
		
		def _to_a3m(sequences: Sequence[str]) -> str:
			names = [f'sequence {i}' for i in range(1, len(sequences)+1)]
			a3m = []
			for sequence, name in zip(sequences, names):
				a3m.append(u'>' + name + u'\n')
				a3m.append(sequence + u'\n')
			return ''.join(a3m)

		print(f'Aligning {len(sequences)} sequences')

		for seq in sequences:
			if len(seq) < 6:
				raise ValueError(f'Kalign: sequences should be at least 6 res long, got {seq}, {len(seq)}')
		with utils.tmpdir_manager() as query_tmp_dir:
			input_fasta_path = tmpdir_manager / Path('input.fasta')
			output_a3m_path = tmpdir_manager / Path('output.a3m')
		
			with open(input_fasta_path, 'w') as f:
				f.write(_to_a3m(sequences))
		
			cmd = [
				self.binary_path.as_posix(),
				'-i', input_fasta_path.as_posix(),
				'-o', output_a3m_path.as_posix(),
				'-format', 'fasta'
			]
		
			print(f"Launching subprocess {''.join(cmd)}")
			process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			with utils.timing(f'Kalign query'):
				stdout, stderr = process.communicate()
				retcode = process.wait()
			if retcode:
				raise RuntimeError(f"Kalign failed:\nstdout:\n{stdout.decode('utf-8')}\nstderr:\n{stderr.decode('utf-8')}")
			
			with open(output_a3m_path) as f:
				a3m = f.read()
		
		return a3m