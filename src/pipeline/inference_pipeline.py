# src/pipeline/inference_pipeline.py

"""
Inference pipeline orchestration.

Coordinates all components to run end-to-end inference:
- Setup environment
- Initialize components
- Execute inference
- Process results
- Save outputs
"""
import json
import time
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from tqdm import tqdm
# === IMPORTS ===
from src.config import Config
from src.utils import (
    setup_reproducible_environment, 
    postprocess_response, 
    save_experiment_results,
    create_sentiment_dataloader
)
from src.model import *
from src.prompt_templates import *


# === MAIN CLASS ===
class InferencePipeline:
    """
    Main inference pipeline orchestrator.
    
    Architecture:
    - Config-driven initialization
    - Stateful execution (tracks progress)
    - Error handling with recovery
    - Special case handling (CoT 2-stage)
    - Uses DataLoader for efficient batching
    """
    
    # === INITIALIZATION ===
    def __init__(self, config: Config):
        """
        Initialize pipeline with config.
        
        Args:
            config: Validated Config object
        """
        self.config = config
        
        # Components (initialized in setup)
        self.model = None
        self.prompt_template = None
        self.dataloader = None
        self.dataset = None
        
        # Results storage
        self.raw_results = []
        self.final_results = []
        
        # Special handling for Zero-shot CoT
        self.reasoning_map = {}  # For stage 2: sent_id -> reasoning
        
        # Statistics tracking
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_time': 0,
            'total_generation_time': 0.0,  
            'batch_times': [],              
            'successful': 0,
            'failed': 0,
            'total_samples': 0,
            'total_opinions': 0,
            'total_input_tokens': 0,      
            'total_output_tokens': 0      
        }
        
        # Flags
        self.is_setup = False
        self.is_completed = False
        
    # === SETUP PHASE (Công đoạn 2-5) ===
    def setup(self):
        """Setup all components."""
        if self.is_setup:
            print("⚠️  Pipeline already setup!")
            return
        
        print("\n" + "=" * 80)
        print("🔧 SETUP PHASE")
        print("=" * 80)
        
        self._setup_environment()
        self._init_prompt_template()
        self._init_model()
        self._load_dataset()
        
        self.is_setup = True
        print("\n✅ Pipeline setup complete!\n")
    
    def _setup_environment(self):
        """Công đoạn 2: Environment setup."""
        print("\n[1/4] Setting up reproducible environment...")
        setup_reproducible_environment(seed=self.config.random_seed)
        print(f"  ✅ Random seed set to {self.config.random_seed}")
    
    def _load_dataset(self):
        """Công đoạn 5: Load data."""
        print("\n[4/4] Loading dataset...")
        
        dataset_path = self.config.data.get_dataset_path()
        print(f"  📁 Dataset: {dataset_path}")
        
        is_two_stage = (self.config.prompt.technique == "zero_shot_cot")

        if not is_two_stage:
            prompt_generator = SingleStagePromptGen(self.prompt_template)
        else:
            prompt_generator = CoTStage1PromptGen(self.prompt_template)

        self.dataloader = create_sentiment_dataloader(
            data_path=dataset_path,
            tokenizer=self.model.tokenizer,
            prompt_generator=prompt_generator,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            chat_template_builder=self.model.chat_template_builder,
            enable_thinking=self.config.model.generation_args.enable_thinking
        )

        if self.config.data.num_samples is not None and self.config.data.num_samples > 0:
            original_len = len(self.dataloader.dataset)
            self.dataloader.dataset.data = self.dataloader.dataset.data[:self.config.data.num_samples]
            print(f"  📊 Limited to {len(self.dataloader.dataset)}/{original_len} samples")
        else:
            print(f"  📊 Loaded {len(self.dataloader.dataset)} samples")

        # Đồng bộ self.dataset để dùng chung
        self.dataset = self.dataloader.dataset.data
        self.stats['total_samples'] = len(self.dataset)
    
    def _init_prompt_template(self):
        """Công đoạn 3: Initialize prompt template."""
        print("\n[2/4] Initializing prompt template...")
        
        technique = self.config.prompt.technique
        print(f"  🎯 Technique: {technique}")
        print(f"  🌐 Language: {'English' if self.config.prompt.language == 'en' else 'Vietnamese'}")
        
        # Get examples pool path if needed
        examples_pool_path = None
        if technique in ["few_shot", "few_shot_cot"]:
            if self.config.data.examples_pool:
                examples_pool_path = self.config.data.get_examples_pool_path()
                print(f"  📚 Examples pool: {examples_pool_path}")
        
        # Create prompt template instance
        self.prompt_template = self._get_prompt_template_instance(
            examples_pool_path=examples_pool_path
        )
        
        # Prepare template (load examples, cache system prompt)
        self.prompt_template.prepare()
        print(f"  ✅ {technique} prompt template ready")
    
    def _init_model(self):
        """Công đoạn 4: Initialize model."""
        print("\n[3/4] Initializing model...")
        
        model_name = self.config.model.name
        print(f"  🤖 Model: {model_name}")
        print(f"  🆔 Model ID: {self.config.model.init_args.model_id}")
        print(f"  📦 Batch size: {self.config.data.batch_size}")
       
        # Create model instance
        self.model = self._get_model_instance()
        
        # Load model
        self.model.load_model()
        print(f"  ✅ {model_name} model ready")
    
    # ========================================================================
    # INFERENCE PHASE
    # ========================================================================
    
    def run(self):
        """Execute inference."""
        if not self.is_setup:
            raise RuntimeError("Pipeline not setup! Call setup() first.")
        
        if self.is_completed:
            print("⚠️  Inference already completed!")
            return
        
        print("\n" + "=" * 80)
        print("🚀 INFERENCE PHASE")
        print("=" * 80)
        
        self.stats['start_time'] = datetime.now().isoformat()
        start_time = time.time()
        
        # Check if Zero-shot CoT (2-stage)
        is_two_stage = (
            self.config.prompt.technique == "zero_shot_cot"
        )
        
        if is_two_stage:
            print("\n📋 Running Two-Stage Inference (Zero-shot CoT)")
            self._run_two_stage_inference()
        else:
            print("\n📋 Running Single-Stage Inference")
            self._run_single_stage_inference()
        
        # Postprocess results
        self._postprocess_results()
        
        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['total_time'] = time.time() - start_time
        self.is_completed = True
        
        print("\n✅ Inference phase complete!\n")
   
    def _run_single_stage_inference(self):
        """
        Single-stage inference using DataLoader prepared in setup().
        
        Uses data_loader infrastructure for efficient batching.
        Leverages pre-tokenized inputs from collator (no re-tokenization).
        Tracks prompts and timing for debug_info.
        """
        if self.dataloader is None:
            raise RuntimeError("Dataloader not prepared. Call setup() first.")

        dataloader = self.dataloader
        total_samples = len(dataloader.dataset)
        num_batches = len(dataloader)
        
        print(f"Processing {total_samples} samples in {num_batches} batches (batch_size={self.config.data.batch_size})...")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference", unit="batch")):
            try:
                texts = batch["texts"]
                sent_ids = batch["sent_ids"]
                system_prompts = batch["system_prompts"]
                user_prompts = batch["user_prompts"]
                
                # === COUNT INPUT TOKENS FROM BATCH ===
                input_token_counts = batch['attention_mask'].sum(dim=1).tolist()
        
                tokenized_inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask']
                }
                
                batch_responses, gen_time, output_token_counts = self.model.generate_batch(tokenized_inputs)
                
                # === STATS ===
                self.stats['total_generation_time'] += gen_time
                self.stats['total_input_tokens'] += sum(input_token_counts)
                self.stats['total_output_tokens'] += sum(output_token_counts)
                self.stats['batch_times'].append({
                    'batch_idx': batch_idx,
                    'batch_size': len(texts),
                    'generation_time': gen_time,
                    'time_per_sample': gen_time / len(texts) if len(texts) > 0 else 0
                })
                
                time_per_sample = gen_time / len(texts) if len(texts) > 0 else 0
                for text, sent_id, sys_p, usr_p, response, in_tok, out_tok in zip(
                    texts, sent_ids, system_prompts, user_prompts, batch_responses,
                    input_token_counts, output_token_counts
                ):
                    self.raw_results.append({
                        'sent_id': int(sent_id),
                        'text': text,
                        'system_prompt': sys_p,
                        'user_prompt': usr_p,
                        'raw_response': response,
                        'generation_time': time_per_sample,
                        'input_tokens': in_tok,      
                        'output_tokens': out_tok,    
                        'success': True
                    })
                    self.stats['successful'] += 1
                
                if (batch_idx + 1) % self.config.cleanup_frequency == 0:
                    torch.cuda.empty_cache()
            
            except Exception as e:
                print(f"\n⚠️  Error processing batch {batch_idx + 1}/{num_batches}: {e}")
                for i in range(len(batch["sent_ids"])):
                    self.raw_results.append({
                        'sent_id': int(batch["sent_ids"][i]),
                        'text': batch["texts"][i],
                        'error': str(e),
                        'success': False
                    })
                    self.stats['failed'] += 1
    
    def _run_two_stage_inference(self):
        """
        Zero-shot CoT two-stage inference.

        Stage 1: Generate reasoning (batched with DataLoader)
        Stage 2: Extract structured output (batched with DataLoader, using preloaded_data)
        """
        batch_size = self.config.data.batch_size

        # ===== STAGE 1: Generate Reasoning (DataLoader) =====
        print("\n--- STAGE 1: Generate Reasoning ---")

        dataloader = self.dataloader
        total_samples = len(dataloader.dataset)
        num_batches = len(dataloader)
        print(f"Processing {total_samples} samples in {num_batches} batches...")

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Stage 1", unit="batch")):
            try:
                # === COUNT INPUT TOKENS FROM BATCH ===
                input_token_counts = batch['attention_mask'].sum(dim=1).tolist()
        
                tokenized_inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask']
                }

                batch_reasonings, gen_time, output_token_counts = self.model.generate_batch(tokenized_inputs)
                
                # === STATS ===
                self.stats['total_generation_time'] += gen_time
                self.stats['total_input_tokens'] += sum(input_token_counts)
                self.stats['total_output_tokens'] += sum(output_token_counts)
                time_per_sample = gen_time / len(batch['sent_ids']) if len(batch['sent_ids']) > 0 else 0

                for sent_id, text, sys_p, usr_p, reasoning, in_tok, out_tok in zip(
                    batch['sent_ids'], batch['texts'], batch['system_prompts'], 
                    batch['user_prompts'], batch_reasonings,
                    input_token_counts, output_token_counts
                ):
                    sid = str(sent_id)
                    self.reasoning_map[sid] = {
                        'text': text,
                        'system_prompt': sys_p,
                        'user_prompt': usr_p,
                        'raw_response': reasoning,
                        'generation_time': time_per_sample,
                        'input_tokens': in_tok,    
                        'output_tokens': out_tok    
                    }

                if (batch_idx + 1) % self.config.cleanup_frequency == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n⚠️  Stage 1 error for batch {batch_idx + 1}/{num_batches}: {e}")
                for sent_id, text in zip(batch['sent_ids'], batch['texts']):
                    sid = str(sent_id)
                    self.reasoning_map[sid] = {
                        'text': text,
                        'system_prompt': '',
                        'user_prompt': '',
                        'raw_response': '',
                        'generation_time': 0.0,
                        'error': str(e)
                    }

        print(f"✅ Stage 1 complete: {len(self.reasoning_map)} reasonings generated")

        # ===== STAGE 2: Extract Structured Output (DataLoader with preloaded_data) =====
        print("\n--- STAGE 2: Extract Structured Output ---")

        stage2_dataloader = create_sentiment_dataloader(
            data_path=None,
            tokenizer=self.model.tokenizer,
            prompt_generator=CoTStage2PromptGen(self.prompt_template, self.reasoning_map),
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            preloaded_data=self.dataset,  # cùng subset/thứ tự như Stage 1
            chat_template_builder=self.model.chat_template_builder,
            enable_thinking=self.config.model.generation_args.enable_thinking
        )

        total_samples = len(stage2_dataloader.dataset)
        num_batches = len(stage2_dataloader)
        print(f"Processing {total_samples} samples in {num_batches} batches...")

        for batch_idx, batch in enumerate(tqdm(stage2_dataloader, desc="Stage 2", unit="batch")):
            try:
                texts = batch["texts"]
                sent_ids = batch["sent_ids"]
                system_prompts = batch["system_prompts"]
                user_prompts = batch["user_prompts"]

                # === COUNT INPUT TOKENS FROM BATCH ===
                input_token_counts = batch['attention_mask'].sum(dim=1).tolist()
        
                tokenized_inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask']
                }
        
                batch_responses, gen_time, output_token_counts = self.model.generate_batch(tokenized_inputs)
                
                # === STATS ===
                self.stats['total_generation_time'] += gen_time
                self.stats['total_input_tokens'] += sum(input_token_counts)
                self.stats['total_output_tokens'] += sum(output_token_counts)
                time_per_sample = gen_time / len(texts) if len(texts) > 0 else 0

                for text, sid, sys_p2, usr_p2, response2, in_tok, out_tok in zip(
                    texts, sent_ids, system_prompts, user_prompts, batch_responses,
                    input_token_counts, output_token_counts
                ):
                    sid_str = str(sid)
                    st1 = self.reasoning_map.get(sid_str, {})
                    if not st1 or not st1.get('raw_response'):
                        raise ValueError(f"No reasoning from Stage 1 for {sid_str}")

                    self.raw_results.append({
                        'sent_id': int(sid_str),
                        'text': text,
                        'stage_1': {
                            'system_prompt': st1.get('system_prompt', ''),
                            'user_prompt': st1.get('user_prompt', ''),
                            'raw_response': st1.get('raw_response', ''),
                            'generation_time': st1.get('generation_time', 0.0),
                            'input_tokens': st1.get('input_tokens', 0),      
                            'output_tokens': st1.get('output_tokens', 0)     
                        },
                        'stage_2': {
                            'system_prompt': sys_p2,
                            'user_prompt': usr_p2,
                            'raw_response': response2,
                            'generation_time': time_per_sample,
                            'input_tokens': in_tok,      
                            'output_tokens': out_tok     
                        },
                        'success': True
                    })
                    self.stats['successful'] += 1

                if (batch_idx + 1) % self.config.cleanup_frequency == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n⚠️  Stage 2 error for batch {batch_idx + 1}/{num_batches}: {e}")
                for i in range(len(batch["sent_ids"])):
                    self.raw_results.append({
                        'sent_id': int(batch["sent_ids"][i]),
                        'text': batch["texts"][i],
                        'error': str(e),
                        'success': False
                    })
                    self.stats['failed'] += 1

        print(f"✅ Stage 2 complete!")
    
    def _postprocess_results(self):
        """Postprocess all raw responses."""
        print("\n📝 Postprocessing results...")
        
        for result in tqdm(self.raw_results, desc="Postprocessing", unit="result"):
            if not result.get('success', False):
                continue
            
            try:
                # Extract raw response (depends on stage)
                if 'stage_2' in result:
                    # Multi-stage: use stage 2 response
                    raw_response = result['stage_2']['raw_response']
                else:
                    # Single-stage
                    raw_response = result['raw_response']
                
                # Extract JSON from raw response
                json_response = self.model.extract_response(raw_response)
                
                # Postprocess to SemEval format
                processed = postprocess_response(
                    response_text=json_response,
                    original_text=result['text'],
                    sent_id=result['sent_id']
                )
                
                # Parse and count opinions
                processed_json = json.loads(processed)
                num_opinions = len(processed_json.get('opinions', []))
                self.stats['total_opinions'] += num_opinions
                
                # Store final result
                self.final_results.append({
                    'sent_id': int(result['sent_id']),
                    'text': result['text'],
                    'result': processed,
                    'num_opinions': num_opinions
                })
                
            except Exception as e:
                print(f"\n⚠️  Postprocessing error for {result.get('sent_id')}: {e}")
                self.stats['failed'] += 1
                self.stats['successful'] -= 1
        
        print(f"✅ Postprocessed {len(self.final_results)} results")
    
    # ========================================================================
    # SAVING PHASE
    # ========================================================================
    
    def save(self):
        """Save results and metadata."""
        if not self.is_completed:
            raise RuntimeError("Inference not completed! Call run() first.")
        
        print("\n" + "=" * 80)
        print("💾 SAVING PHASE")
        print("=" * 80)
        
        # Calculate statistics
        self._calculate_statistics()
        
        # Prepare results for saving
        results_to_save = []
        for result in self.final_results:
            results_to_save.append({
                'sent_id': result['sent_id'],
                'text': result['text'],
                **json.loads(result['result'])
            })
        
        # Save using utility function
        print("\n📁 Saving experiment results...")
        saved_paths = save_experiment_results(
            results=results_to_save,
            config=self.config,
            statistics=self.stats,
            raw_results=self.raw_results 
        )
        
        print("\n✅ Results saved:")
        for key, path in saved_paths.items():
            if path:
                print(f"  📄 {key}: {path}")
    
    # ========================================================================
    # CLEANUP PHASE
    # ========================================================================
    
    def cleanup(self):
        """Cleanup and summary."""
        print("\n" + "=" * 80)
        print("🧹 CLEANUP PHASE")
        print("=" * 80)
        
        # Show summary
        self._show_summary()
        
        # Unload model
        if self.model:
            print("\n🗑️  Unloading model...")
            self.model.cleanup()
            print("  ✅ Model unloaded")
        
        print("\n" + "=" * 80)
        print("✨ PIPELINE COMPLETE!")
        print("=" * 80)
    
    def _show_summary(self):
        """Display summary statistics."""
        print("\n📊 EXPERIMENT SUMMARY")
        print("-" * 80)
        print(f"Experiment     : {self.config.experiment.name}")
        print(f"Model          : {self.config.model.name}")
        print(f"Prompt         : {self.config.prompt.technique}")
        print(f"Language       : {'English' if self.config.prompt.language == 'en' else 'Vietnamese'}")
        print(f"Batch Size     : {self.config.data.batch_size}")
        print("-" * 80)
        print(f"Total Samples  : {self.stats['total_samples']}")
        print(f"Successful     : {self.stats['successful']}")
        print(f"Failed         : {self.stats['failed']}")
        print(f"Success Rate   : {self.stats.get('success_rate', 0):.2f}%")
        print(f"Total Opinions : {self.stats['total_opinions']}")
        print(f"Avg Opinions   : {self.stats.get('avg_opinions_per_sample', 0):.2f}")
        print("-" * 80)
        print(f"Total Time     : {self.stats['total_time']:.2f}s")
        print(f"Generation Time: {self.stats['total_generation_time']:.2f}s")
        print(f"Avg Time/Sample: {self.stats.get('avg_sample_time', 0):.2f}s")
        print(f"Avg Batch Time : {self.stats.get('avg_batch_time', 0):.2f}s")
        print("-" * 80)
        print(f"Total Input Tokens      : {self.stats['total_input_tokens']:,}")
        print(f"Total Output Tokens     : {self.stats['total_output_tokens']:,}")
        print(f"Total Tokens            : {self.stats.get('total_tokens', 0):,}")
        print(f"Avg Input Tokens/Sample : {self.stats.get('avg_input_tokens_per_sample', 0):.2f}")
        print(f"Avg Output Tokens/Sample: {self.stats.get('avg_output_tokens_per_sample', 0):.2f}")
        print("-" * 80)
    
    # ========================================================================
    # MAIN ENTRY
    # ========================================================================
    
    def execute(self):
        """Execute complete pipeline."""
        try:
            self.setup()
            self.run()
            self.save()
            self.cleanup()
        except KeyboardInterrupt:
            print("\n\n⚠️  Pipeline interrupted by user!")
            if self.model:
                print("🗑️  Cleaning up model...")
                self.model.cleanup()
            raise
        except Exception as e:
            print(f"\n\n❌ Pipeline failed: {e}")
            if self.model:
                print("🗑️  Cleaning up model...")
                self.model.cleanup()
            raise
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _get_prompt_template_instance(
        self,
        examples_pool_path: Optional[str] = None
    ):
        """Factory method for prompt templates."""
        technique = self.config.prompt.technique
        eng = True if self.config.prompt.language == 'en' else False
        
        if technique == "few_shot":
            return FewShotPrompt(
                eng=eng,
                n_shot=self.config.prompt.n_shot,
                examples_pool_path=examples_pool_path
            )
        
        elif technique == "few_shot_cot":
            return FewShotCoTPrompt(
                eng=eng,
                n_shot=self.config.prompt.n_shot
            )
        
        elif technique == "zero_shot_cot":
            return ZeroShotCoTPrompt(eng=eng)
        
        elif technique == "rereading":
            return ReReadingPrompt(
                eng=eng,
                add_method=self.config.prompt.add_method
            )
        
        elif technique == "plan_and_solve":
            return PlanAndSolvePrompt(
                eng=eng,
                plus=self.config.prompt.plus_mode
            )
        
        else:
            raise ValueError(f"Unknown prompt technique: {technique}")
    
    def _get_model_instance(self):
        """Factory method for models."""
        model_name = self.config.model.name
        
        if model_name == "gemma":
            return GemmaModel(config=self.config.model)
        elif model_name == "qwen":
            return QwenModel(config=self.config.model)
        elif model_name == "seallm":
            return SeaLLModel(config=self.config.model)
        elif model_name == "vistral":
            return VistralModel(config=self.config.model)
        elif model_name == "llama4":
            return Llama4Model(config=self.config.model)
        elif model_name == "llama3":
            return Llama3Model(config=self.config.model)
        elif model_name == "vinallama":
            return VinallamaModel(config=self.config.model)
        # elif model_name == "mistral":
        #     return MistralModel(config=self.config.model)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _calculate_statistics(self):
        """Calculate pipeline statistics."""
        total = self.stats['total_samples']
        successful = self.stats['successful']
        
        self.stats['success_rate'] = (successful / total * 100) if total > 0 else 0
        self.stats['avg_opinions_per_sample'] = (
            self.stats['total_opinions'] / successful if successful > 0 else 0
        )
        self.stats['avg_time_per_sample'] = (
            self.stats['total_time'] / total if total > 0 else 0
        )
        
        self.stats['avg_input_tokens_per_sample'] = (
            self.stats['total_input_tokens'] / successful if successful > 0 else 0
        )
        self.stats['avg_output_tokens_per_sample'] = (
            self.stats['total_output_tokens'] / successful if successful > 0 else 0
        )
        self.stats['total_tokens'] = (
            self.stats['total_input_tokens'] + self.stats['total_output_tokens']
        )
        self.stats['avg_batch_time'] = (
            sum(b['generation_time'] for b in self.stats['batch_times']) / len(self.stats['batch_times'])
            if self.stats['batch_times'] else 0
        )