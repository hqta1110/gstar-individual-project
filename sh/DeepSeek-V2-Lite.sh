# sh sh/DeepSeek-V2-Lite.sh 2>&1 | tee logs/DeepSeek-V2-Lite.log

echo 'Profiling Down_proj Outliers of DeepSeek-V2-Lite'
CUDA_VISIBLE_DEVICES=6 python3 run.py \
--model_name Your Model Path \
--save_path ./output/DeepSeek-V2-Lite \
--profile_outliers \
--vis_outliers_heatmap

echo 'Profiling Massive Experts of DeepSeek-V2-Lite'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/DeepSeek-V2-Lite \
--profile_massive_experts \
--vis_massive_experts_line_plot

echo 'Profiling Super Experts of DeepSeek-V2-Lite'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/DeepSeek-V2-Lite \
--profile_super_experts \
--vis_super_experts_line_plot

echo 'PPL Test of Original DeepSeek-V2-Lite'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/DeepSeek-V2-Lite \
--dataset_name wikitext2 \
--eval_ppl

echo 'PPL Test of DeepSeek-V2-Lite After Rrune Experts'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/DeepSeek-V2-Lite \
--eval_ppl \
--dataset wikitext2 \
--prune_experts "3,54;4,38;2,3;5,63"

echo 'PPL Test of DeepSeek-V2-Lite After Rrune Super Experts'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/DeepSeek-V2-Lite \
--eval_ppl \
--dataset wikitext2 \
--prune_super_experts
