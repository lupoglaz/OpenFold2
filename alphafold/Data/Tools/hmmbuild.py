from pathlib import Path
from alphafold.Data.Tools import utils
import re

class HMMBuild:
	def __init__(self,
				binary_path: Path,
				singlemx: bool=False):
		self.binary_path = binary_path
		self.singlemx = singlemx


	def _build_profile(self, msa: str, model_construction: str='fast') -> str:
		if model_construction not in {'hand', 'fast'}:
			raise ValueError(f'HMMBuild: invalid model construction {model_construction}')
		with utils.tmpdir_manager() as query_tmp_dir:
			input_query = query_tmp_dir / Path('query.msa')
			output_hmm_path = query_tmp_dir / Path('output.hmm')
			with open(input_query, 'w') as f:
				f.write(msa)

			cmd = [self.binary_path.as_posix()]
			if model_construction == 'hand':
				cmd += [f'--{model_construction}']
			if singlemx:
				cmd += ['--singlemx']
			cmd += ['--amino', output_hmm_path.as_posix(), input_query.as_posix()]

			print(f"Launching subprocess {''.join(cmd)}")
			process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			with utils.timing(f'HMMBuild query'):
				stdout, stderr = process.communicate()
				retcode = process.wait()
			if retcode:
				raise RuntimeError(f"HMMBuild failed:\nstdout:\n{stdout.decode('utf-8')}\nstderr:\n{stderr.decode('utf-8')}")
			with open(output_hmm_path, encoding='utf-8') as f:
				hmm = f.read()
		return hmm

	def build_profile_from_a3m(self, a3m: str) -> str:
		lines = []
		for line in a3m.splitlines():
			if not line.startwith('>'):
				line = re.sub('[a-z]+', '', line)
			lines.append(line + '\n')
		msa = ''.join(lines)
		return self._build_profile(msa, model_construction='fast')

	def build_profile_from_sto(self, sto: str, model_construction: str='fast') -> str:
		return self._build_profile(sto, model_construction=model_construction)
