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
        - Do not return string values such as "CUDA" — always return the full key-value pair.

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
        genai.configure(api_key=self.api_key)

    def count_tokens(self, prompt, model_name="gemini-2.0-flash-exp"):
        model = genai.GenerativeModel(model_name)
        try:
            result = model.count_tokens(prompt)
            return result.total_tokens
        except Exception as e:
            print(f"Error counting tokens: {e}")
            return -1

    def collect_docs(self, docs_dir):
        docs_content = ""
        for root, _, files in os.walk(docs_dir):
            for file in sorted(files):
                if file.lower().endswith((".md", ".txt", ".rst")):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            rel_path = os.path.relpath(file_path, docs_dir)
                            docs_content += f"\n\n# {rel_path}\n" + f.read()
                    except Exception as e:
                        print(f"Skipping {file_path}: {e}")
        return docs_content

    def build_full_prompt(self, docs_path, options, features):
        docs_content = self.collect_docs(docs_path)
        return self.PROMPTS["gromacs_automated_selection"].format(
            docs=docs_content,
            options=json.dumps(options, indent=2),
            features=json.dumps(features, indent=2)
        )
    
    def count_install_guide_tokens(self, docs_dir, model_name="gemini-2.0-flash-exp"):
        install_dir = os.path.join(docs_dir, "install-guide")
        if not os.path.isdir(install_dir):
            print(f"⚠️ install-guide directory not found in {docs_dir}")
            return -1
        docs_content = self.collect_docs(install_dir)
        prompt = self.PROMPTS["gromacs_automated_selection"].format(
            docs=docs_content,
            options="{}",
            features="{}"
        )
        return self.count_tokens(prompt, model_name)

if __name__ == "__main__":
    docs_path = "./docs"
    options_path = "./options.json"
    features_path = "./system-features.json"

    with open(options_path, 'r') as f:
        options = json.load(f)
    with open(features_path, 'r') as f:
        features = json.load(f)

    helper = GeminiInterface()
    full_prompt = helper.build_full_prompt(docs_path, options, features)

    with open("full_prompt.txt", "w", encoding="utf-8") as f:
        f.write(full_prompt)

    token_count = helper.count_tokens(full_prompt)
    print(f"\n✅ Total token count for full prompt: {token_count}")

    install_token_count = helper.count_install_guide_tokens(docs_path)
    print(f"📦 Token count for install-guide content only: {install_token_count}")