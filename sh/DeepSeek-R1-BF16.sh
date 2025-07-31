# sh sh/DeepSeek-R1-BF16.sh 2>&1 | tee logs/DeepSeek-R1-BF16.log

echo 'Profiling Down_proj Outliers of DeepSeek-R1-BF16'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/DeepSeek-R1-BF16 \
--profile_outliers \
--vis_outliers_heatmap

echo 'Profiling Massive Experts of DeepSeek-R1-BF16'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/DeepSeek-R1-BF16 \
--profile_massive_experts \
--vis_massive_experts_line_plot

echo 'Profiling Super Experts of DeepSeek-R1-BF16'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/DeepSeek-R1-BF16 \
--profile_super_experts \
--vis_super_experts_line_plot

echo 'PPL Test of Original DeepSeek-R1-BF16'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/DeepSeek-R1-BF16 \
--dataset_name wikitext2 \
--eval_ppl

echo 'PPL Test of DeepSeek-R1-BF16 After Rrune Experts'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/DeepSeek-R1-BF16 \
--eval_ppl \
--dataset wikitext2 \
--prune_experts "3,54;4,38;2,3;5,63"

echo 'PPL Test of DeepSeek-R1-BF16 After Rrune Super Experts'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/DeepSeek-R1-BF16 \
--eval_ppl \
--dataset wikitext2 \
--prune_super_experts
