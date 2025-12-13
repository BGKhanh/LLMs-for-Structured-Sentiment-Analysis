from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from typing import List, Tuple, Dict, Any
import time

class VLLMModel:
    """
    Unified vLLM model wrapper.
    Replaces all individual model classes (Gemma, Qwen, Llama, etc.)
    """
    
    def __init__(self, config):
        self.config = config
        self.llm = None
        self.is_loaded = False
    
    def load_model(self):
        # Extract init_args
        kwargs = self.config.init_args.to_dict()
        model_id = kwargs.pop("model_id")
        
        # Create vLLM instance
        self.llm = LLM(
            model=model_id,
            **kwargs  # vLLM auto-handles most params
        )
        self.is_loaded = True
    
    def generate_batch(self, prompts: List[str]) -> Tuple[List[str], float, List[int], List[int]]:
        # Create SamplingParams from generation_args
        gen_kwargs = self.config.generation_args.to_dict()
        sampling_params = SamplingParams(**gen_kwargs)
        
        # Generate
        start = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        gen_time = time.time() - start
        
        # Extract results
        responses = [out.outputs[0].text for out in outputs]
        input_tokens = [len(out.prompt_token_ids) for out in outputs]
        output_tokens = [len(out.outputs[0].token_ids) for out in outputs]
        
        return responses, gen_time, output_tokens, input_tokens
    
    def cleanup(self) -> None:
            """
            Clean up vLLM GPU memory properly.
            Quy trình bắt buộc: Xóa Object -> Gọi GC -> Xóa Cache.
            """
            print("🧹 Starting vLLM cleanup...")

            # 1. Hủy object vLLM để giải phóng workers
            if hasattr(self, 'llm'):
                del self.llm
                self.llm = None
            
            # 2. Xóa các tham số khác nếu có
            if hasattr(self, 'sampling_params'):
                del self.sampling_params
            
            # 3. Ép Python thu gom rác (Bắt buộc để trigger __del__ của vLLM engine)
            gc.collect()

            # 4. Dọn dẹp PyTorch Cache (Code cũ của bạn)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Nếu dùng thư viện Ray (vLLM thường dùng Ray cho multi-gpu)
                # import ray
                # if ray.is_initialized():
                #     ray.shutdown()

                # Dọn dẹp IPC (cho multi-processing)
                try:
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                except Exception:
                    pass # Bỏ qua lỗi nếu không truy cập được device khác
                    
            print("✨ GPU memory cleanup completed!")
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        model_id = self.config.init_args.model_id
        return {
            "model_id": model_id,
            "is_loaded": self.is_loaded,
            "framework": "vLLM"
        }