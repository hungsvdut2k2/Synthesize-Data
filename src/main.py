import argparse
import json
import os.path
import time

from datasets import load_dataset
from tqdm import tqdm

from src.schemas import SynthesizeParameters
from src.settings import settings
from src.synthesizer import Synthesizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--end-index", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prompt-file-path", type=str, required=True)
    parser.add_argument("--time-sleep", type=int, default=5)
    args = parser.parse_args()


    persona_dataset = load_dataset("proj-persona/PersonaHub", "persona")["train"]["persona"]
    persona_dataset = persona_dataset[args.start_index : args.end_index]

    system_prompt = json.load(open(args.prompt_file_path))["system_prompt"]

    synthesizer = Synthesizer(
        google_cookies_dir=settings.GOOGLE_COOKIES_DIR,
        deepinfra_tokens_dir=settings.DEEPINFRA_API_KEY_DIR,
        huggingface_tokens_dir=settings.HUGGINGFACE_TOKEN_DIR,
    )


    for index, row in tqdm(enumerate(persona_dataset), total=len(persona_dataset)):
        synthesizer_parameters = SynthesizeParameters(
            system_prompt=system_prompt,
            user_prompt=row,
        )
        result = synthesizer.synthesize(synthesizer_parameters)

        with open(os.path.join(args.output_dir, "{}.txt".format(index)), "w") as f:
            f.write(result)

        time.sleep(args.time_sleep)
