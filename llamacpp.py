#!/usr/bin/env python3
"""
CodeLlama Inference using llama_cpp
This script performs inference on a CodeLlama model using the llama_cpp Python bindings
with parameters optimized for code generation.
"""

import argparse
import os
from typing import Dict, Any, Optional
from llama_cpp import Llama

class CodeLlamaInference:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0
    ):
        """
        Initialize CodeLlama model with llama_cpp
        
        Args:
            model_path: Path to the CodeLlama GGUF model file
            n_ctx: Context size for the model (in tokens)
            n_threads: Number of threads to use (default: auto-detect)
            n_gpu_layers: Number of layers to offload to GPU (0 for CPU only)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if n_threads is None:
            # Auto-detect the number of threads
            import multiprocessing
            n_threads = max(1, multiprocessing.cpu_count() - 1)
        
        print(f"Loading model from {model_path}...")
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers
        )
        print(f"Model loaded successfully with {n_threads} threads.")
    
    def generate_code(
        self,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        max_tokens: int = 2048,
        repeat_penalty: float = 1.1,
        stop: list = ["```", "</code>", "Human:", "User:"]
    ) -> Dict[str, Any]:
        """
        Generate code from a prompt with optimized parameters for code generation
        
        Args:
            prompt: Input prompt for code generation
            temperature: Randomness of generation (lower is more deterministic)
            top_p: Nucleus sampling parameter
            top_k: Limit vocabulary to top k tokens
            max_tokens: Maximum number of tokens to generate
            repeat_penalty: Penalty for repeating tokens
            stop: Tokens that stop generation
            
        Returns:
            Dictionary containing the generated text and other metadata
        """
        # Format prompt for code generation (optional)
        formatted_prompt = f"Generate code for the following task:\n\n{prompt}\n\n```python\n"
        
        # Generate response
        response = self.model(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop,
            echo=False
        )
        
        return response
    
    def interactive_mode(self):
        """Enter interactive mode for multiple prompts"""
        print("\nCodeLlama Interactive Mode")
        print("Enter 'exit', 'quit', or Ctrl+C to exit\n")
        
        try:
            while True:
                user_prompt = input("\nEnter your prompt: ")
                if user_prompt.lower() in ['exit', 'quit']:
                    break
                
                print("\nGenerating code...\n")
                result = self.generate_code(user_prompt)
                
                # Extract and clean up the generated code
                generated_code = result['choices'][0]['text']
                print("\n--- Generated Code ---\n")
                print(generated_code)
                print("\n---------------------\n")
                
        except KeyboardInterrupt:
            print("\nExiting interactive mode")


def main():
    parser = argparse.ArgumentParser(description="CodeLlama Inference using llama_cpp")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to the CodeLlama GGUF model file"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Prompt for code generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for text generation (default: 0.2)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k sampling parameter (default: 40)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate (default: 2048)"
    )
    parser.add_argument(
        "--n_ctx",
        type=int,
        default=4096,
        help="Context window size (default: 4096)"
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        help="Number of threads to use (default: auto-detect)"
    )
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=0,
        help="Number of layers to offload to GPU (default: 0, CPU only)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive mode for multiple prompts"
    )
    
    args = parser.parse_args()
    
    inference = CodeLlamaInference(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers
    )
    
    if args.interactive:
        inference.interactive_mode()
    elif args.prompt:
        result = inference.generate_code(
            prompt=args.prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens
        )
        print(result['choices'][0]['text'])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
