lm_eval --model hf \
    --model_args pretrained=google/gemma-2b-it \
    --tasks semeval_vi_fewshot \
    --num_fewshot 5 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --system_instruction "Hãy phân tích cảm xúc của văn bản tiếng Việt sau theo định dạng JSON. Cần trích xuất các opinions bao gồm: Source, Target, Polar_expression, Polarity, Intensity."