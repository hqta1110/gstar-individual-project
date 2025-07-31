# sh sh/Qwen3-30B-A3B.sh 2>&1 | tee logs/Qwen3-30B-A3B.log

echo 'Profiling Down_proj Outliers of Qwen3-30B-A3B'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/Qwen3-30B-A3B \
--profile_outliers \
--vis_outliers_heatmap

echo 'Profiling Massive Experts of Qwen3-30B-A3B'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/Qwen3-30B-A3B \
--profile_massive_experts \
--vis_massive_experts_line_plot

echo 'Profiling Super Experts of Qwen3-30B-A3B'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/Qwen3-30B-A3B \
--profile_super_experts \
--vis_super_experts_line_plot

echo 'PPL Test of Original Qwen3-30B-A3B'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/Qwen3-30B-A3B \
--dataset_name wikitext2 \
--eval_ppl

echo 'PPL Test of Qwen3-30B-A3B After Rrune Experts'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/Qwen3-30B-A3B \
--eval_ppl \
--dataset wikitext2 \
--prune_experts "3,54;4,38;2,3;5,63"

echo 'PPL Test of Qwen3-30B-A3B After Rrune Super Experts'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/Qwen3-30B-A3B \
--eval_ppl \
--dataset wikitext2 \
--prune_super_experts
