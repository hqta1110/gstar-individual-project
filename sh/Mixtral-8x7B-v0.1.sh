# sh sh/Mixtral-8x7B-v0.1.sh 2>&1 | tee logs/Mixtral-8x7B-v0.1.log

echo 'Profiling Down_proj Outliers of Mixtral-8x7B-v0.1'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/Mixtral-8x7B-v0.1 \
--profile_outliers \
--vis_outliers_heatmap

echo 'Profiling Massive Experts of Mixtral-8x7B-v0.1'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/Mixtral-8x7B-v0.1 \
--profile_massive_experts \
--vis_massive_experts_line_plot

echo 'Profiling Super Experts of Mixtral-8x7B-v0.1'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/Mixtral-8x7B-v0.1 \
--profile_super_experts \
--vis_super_experts_line_plot

echo 'PPL Test of Original Mixtral-8x7B-v0.1'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/Mixtral-8x7B-v0.1 \
--dataset_name wikitext2 \
--eval_ppl

echo 'PPL Test of Mixtral-8x7B-v0.1 After Rrune Experts'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/Mixtral-8x7B-v0.1 \
--eval_ppl \
--dataset wikitext2 \
--prune_experts "3,54;4,38;2,3;5,63"

echo 'PPL Test of Mixtral-8x7B-v0.1 After Rrune Super Experts'
CUDA_VISIBLE_DEVICES=0 python3 run.py \
--model_name Your Model Path \
--save_path ./output/Mixtral-8x7B-v0.1 \
--eval_ppl \
--dataset wikitext2 \
--prune_super_experts
