import google.generativeai as genai
import os
import json


class GeminiInterface:

    PROMPTS = {
        "gromacs_automated_selection": """
        You are an expert in high-performance computing (HPC) software optimization.

        I will provide you with:
        1. Official documentation from the GROMACS project.
        2. A JSON object listing all available build-time specialization options.
        3. A JSON describing all the features of the target system.

        Your task is to select the best combination of build options for maximizing performance on the target system using the content of documentation files as a guide.

        Return a JSON object in the following format (do not include any other text):

        {{
        "vectorization_flags": {{}},
        "gpu_backends": {{}},
        "parallel_libraries": {{}},
        "fft_libraries": {{}},
        "linear_algebra_libraries": {{}},
        "optimization_build_flags": []
        }}

        Guidelines:
        - Prefer SIMD flags like `AVX_512` if supported and mentioned as optimized in the documentation.
        - Use GPU acceleration (e.g., `CUDA`, `SYCL`) if supported and recommended.
        - Choose parallelization options like `MPI`, `OpenMP`, or `Thread-MPI` based on GROMACS guidance.
        - Select FFT and BLAS libraries known for performance (e.g., `fftw3`, `MKL`).
        - Include any GROMACS-specific performance optimization flags mentioned in the docs.
        - Do not invent flags. Only use what's listed in the available options.
        - For each selected option (e.g., CUDA, OpenMP, MKL), return its full dictionary entry from the provided options, including fields like `build_flag`, `version`, and `used_as_default` if present.
        - Do not return string values such as \"CUDA\" — always return the full key-value pair.

        GROMACS Documentation:
        {docs}

        Available Specialization Options:
        {options}

        System Features:
        {features}
        """
    }

    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise EnvironmentError("Error: GOOGLE_API_KEY environment variable not set.")

    def query_gemini(self, prompt, model_name="gemini-2.0-flash-exp"):
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(model_name)

        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()

            if not response_text:
                return {"error": "Empty response from Gemini API"}

            if response_text.startswith("```json"):
                response_text = response_text.split("\n", 1)[1]
            if response_text.endswith("```"):
                response_text = response_text.rsplit("\n", 1)[0]

            return response_text.strip()
        except Exception as e:
            return {"error": str(e)}

    def select_options(self, options, system_features, docs_dir):
        docs_content = ""
        if not os.path.isdir(docs_dir):
            raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

        for file in sorted(os.listdir(docs_dir)):
            doc_path = os.path.join(docs_dir, file)
            if os.path.isfile(doc_path) and file.lower().endswith((".md", ".txt", ".rst")):
                with open(doc_path, 'r', encoding='utf-8') as f:
                    docs_content += f"\n\n# {file}\n" + f.read()

        prompt = self.PROMPTS["gromacs_automated_selection"].format(
            docs=docs_content,
            options=json.dumps(options, indent=2),
            features=json.dumps(system_features, indent=2)
        )

        # Save the prompt to a file for inspection
        with open("debug_gemini_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        response_text = self.query_gemini(prompt)

        if isinstance(response_text, dict) and "error" in response_text:
            raise ValueError(f"Gemini API Error: {response_text['error']}")

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            raise ValueError(f"Error parsing Gemini response as JSON:\n{response_text}")


if __name__ == "__main__":
    # Define input paths
    docs_path = "./docs"
    options_path = "./options.json"
    features_path = "./system-features.json"

    with open(options_path, 'r') as f:
        options = json.load(f)
    with open(features_path, 'r') as f:
        system_features = json.load(f)

    helper = GeminiInterface()
    selected = helper.select_options(options, system_features, docs_path)
    print(json.dumps(selected, indent=2))

