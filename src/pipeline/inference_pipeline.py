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
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from tqdm import tqdm
# === IMPORTS ===
from src.config import Config
from src.utils import (
    setup_reproducible_environment, 
    postprocessing, 
    save_experiment_results
)
from src.model import GemmaModel
from src.prompt_templates import *
# etc.

# === MAIN CLASS ===
class InferencePipeline:
    """
    Main inference pipeline orchestrator.
    
    Architecture:
    - Config-driven initialization
    - Stateful execution (tracks progress)
    - Error handling with recovery
    - Special case handling (CoT 2-stage)
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
            'successful': 0,
            'failed': 0,
            'total_samples': 0,
            'total_opinions': 0
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
        self._load_dataset()
        self._init_prompt_template()
        self._init_model()
        
        self.is_setup = True
        print("\n✅ Pipeline setup complete!\n")
    
    def _setup_environment(self):
        """Công đoạn 2: Environment setup."""
        print("\n[1/4] Setting up reproducible environment...")
        setup_reproducible_environment(seed=self.config.random_seed)
        print(f"  ✅ Random seed set to {self.config.random_seed}")
    
    def _load_dataset(self):
        """Công đoạn 3: Load data."""
        print("\n[2/4] Loading dataset...")
        
        dataset_path = self.config.data.get_dataset_path()
        print(f"  📁 Dataset: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        # Apply n_sample limit if specified
        if self.config.data.n_sample > 0:
            original_len = len(self.dataset)
            self.dataset = self.dataset[:self.config.data.n_sample]
            print(f"  📊 Limited to {len(self.dataset)}/{original_len} samples")
        else:
            print(f"  📊 Loaded {len(self.dataset)} samples")
        
        self.stats['total_samples'] = len(self.dataset)
    
    def _init_prompt_template(self):
        """Công đoạn 4: Initialize prompt template."""
        print("\n[3/4] Initializing prompt template...")
        
        technique = self.config.prompt.technique
        print(f"  🎯 Technique: {technique}")
        print(f"  🌐 Language: {'English' if self.config.prompt.eng else 'Vietnamese'}")
        
        # Get examples pool path if needed
        examples_pool_path = None
        if technique in ["few_shot", "few_shot_cot"]:
            if self.config.data.examples_pool_path:
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
        """Công đoạn 5: Initialize model."""
        print("\n[4/4] Initializing model...")
        
        model_name = self.config.model.name
        print(f"  🤖 Model: {model_name}")
        print(f"  🆔 Model ID: {self.config.model.model_id}")
        
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
        
        self.stats['start_time'] = datetime.now()
        start_time = time.time()
        
        # Check if Zero-shot CoT (2-stage)
        is_two_stage = (
            self.config.prompt.technique == "zero_shot_cot" and
            self.config.prompt.stage == "both"
        )
        
        if is_two_stage:
            print("\n📋 Running Two-Stage Inference (Zero-shot CoT)")
            self._run_two_stage_inference()
        else:
            print("\n📋 Running Single-Stage Inference")
            self._run_single_stage_inference()
        
        # Postprocess results
        self._postprocess_results()
        
        self.stats['end_time'] = datetime.now()
        self.stats['total_time'] = time.time() - start_time
        self.is_completed = True
        
        print("\n✅ Inference phase complete!\n")
   
    def _run_single_stage_inference(self):
        """Normal single-stage inference."""
        print(f"Processing {len(self.dataset)} samples...")
        
        # Process with progress bar
        for sample in tqdm(self.dataset, desc="Inference", unit="sample"):
            try:
                # Get prompts
                text = sample['text']
                sent_id = sample['sent_id']
                
                # For Zero-shot CoT single stage
                kwargs = {}
                if self.config.prompt.technique == "zero_shot_cot":
                    kwargs['stage'] = self.config.prompt.stage
                
                system_prompt, user_prompt = self.prompt_template.get_prompt(
                    text, sent_id, **kwargs
                )
                
                # Generate
                raw_response = self.model.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
                
                # Store result
                self.raw_results.append({
                    'sent_id': sent_id,
                    'text': text,
                    'system_prompt': system_prompt,
                    'user_prompt': user_prompt,
                    'raw_response': raw_response,
                    'success': True
                })
                
                self.stats['successful'] += 1
                
            except Exception as e:
                print(f"\n⚠️  Error processing {sample.get('sent_id', 'unknown')}: {e}")
                self.raw_results.append({
                    'sent_id': sample.get('sent_id', 'unknown'),
                    'text': sample.get('text', ''),
                    'error': str(e),
                    'success': False
                })
                self.stats['failed'] += 1
    
    def _run_two_stage_inference(self):
        """Zero-shot CoT two-stage inference."""
        print("\n--- STAGE 1: Generate Reasoning ---")
        print(f"Processing {len(self.dataset)} samples...")
        
        # Stage 1: Generate reasoning
        stage1_results = []
        for sample in tqdm(self.dataset, desc="Stage 1", unit="sample"):
            try:
                text = sample['text']
                sent_id = sample['sent_id']
                
                system_prompt, user_prompt = self.prompt_template.get_prompt(
                    text, sent_id, stage="stage_1"
                )
                
                reasoning = self.model.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
                
                # Store reasoning for stage 2
                self.reasoning_map[sent_id] = reasoning
                
                stage1_results.append({
                    'sent_id': sent_id,
                    'text': text,
                    'stage': 'stage_1',
                    'system_prompt': system_prompt,
                    'user_prompt': user_prompt,
                    'reasoning': reasoning,
                    'success': True
                })
                
            except Exception as e:
                print(f"\n⚠️  Stage 1 error for {sample.get('sent_id')}: {e}")
                self.reasoning_map[sample['sent_id']] = ""
                stage1_results.append({
                    'sent_id': sample.get('sent_id'),
                    'error': str(e),
                    'success': False
                })
        
        print(f"✅ Stage 1 complete: {len(self.reasoning_map)} reasonings generated")
        
        # Stage 2: Extract structured output
        print("\n--- STAGE 2: Extract Structured Output ---")
        print(f"Processing {len(self.dataset)} samples...")
        
        for sample in tqdm(self.dataset, desc="Stage 2", unit="sample"):
            try:
                text = sample['text']
                sent_id = sample['sent_id']
                reasoning = self.reasoning_map.get(sent_id, "")
                
                if not reasoning:
                    raise ValueError("No reasoning from Stage 1")
                
                system_prompt, user_prompt = self.prompt_template.get_prompt(
                    text, sent_id, stage="stage_2", reasoning=reasoning
                )
                
                raw_response = self.model.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
                
                # Store final result
                self.raw_results.append({
                    'sent_id': sent_id,
                    'text': text,
                    'stage_1_reasoning': reasoning,
                    'stage_2_system_prompt': system_prompt,
                    'stage_2_user_prompt': user_prompt,
                    'raw_response': raw_response,
                    'success': True
                })
                
                self.stats['successful'] += 1
                
            except Exception as e:
                print(f"\n⚠️  Stage 2 error for {sample.get('sent_id')}: {e}")
                self.raw_results.append({
                    'sent_id': sample.get('sent_id', 'unknown'),
                    'text': sample.get('text', ''),
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
                # Extract JSON from raw response
                json_response = self.model.extract_response(result['raw_response'])
                
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
                    'sent_id': result['sent_id'],
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
            statistics=self.stats
        )
        
        print("\n✅ Results saved:")
        for key, path in saved_paths.items():
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
        print(f"Language       : {'English' if self.config.prompt.eng else 'Vietnamese'}")
        print("-" * 80)
        print(f"Total Samples  : {self.stats['total_samples']}")
        print(f"Successful     : {self.stats['successful']}")
        print(f"Failed         : {self.stats['failed']}")
        print(f"Success Rate   : {self.stats['success_rate']:.2f}%")
        print(f"Total Opinions : {self.stats['total_opinions']}")
        print(f"Avg Opinions   : {self.stats['avg_opinions_per_sample']:.2f}")
        print("-" * 80)
        print(f"Total Time     : {self.stats['total_time']:.2f}s")
        print(f"Avg Time/Sample: {self.stats['avg_time_per_sample']:.2f}s")
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
        eng = self.config.prompt.eng
        
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
            return ReReadingPrompt(eng=eng)
        
        elif technique == "plan_and_solve":
            return PlanAndSolvePrompt(
                eng=eng,
                plus=self.config.prompt.plus
            )
        
        else:
            raise ValueError(f"Unknown prompt technique: {technique}")
    
    def _get_model_instance(self):
        """Factory method for models."""
        model_name = self.config.model.name
        
        if model_name == "gemma":
            return GemmaModel(config=self.config.model)
        
        # Add other models here
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