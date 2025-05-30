set -x
[ -z "${MODEL_PATH}" ] && MODEL_PATH=Qwen/Qwen2.5-7B-Instruct-1M
[ -z "${EXP_NAME}" ] && EXP_NAME=Qwen2.5-7B-Instruct-1M-GRPO
[ -z "${SAVE_DIR}" ] && SAVE_DIR=~
[ -z "${n_gpus_per_node}" ] && n_gpus_per_node=4
[ -z "${train_files}" ] && train_files=data/kk/instruct/3ppl/train.parquet
[ -z "${val_files}" ] && val_files=data/kk/instruct/3ppl/test.parquet
[ -z "${max_response_length}" ] && max_response_length=4096
[ -z "${temperature}" ] && temperature=1 
[ -z "${rollout_n}" ] && rollout_n=8 
[ -z "${lr}" ] && lr=4e-7 
[ -z "${train_batch_size}" ] && train_batch_size=4 
[ -z "${project_name}" ] && project_name=GRPO_logic_KK_test
[ -z "${entropy}" ] && entropy=False
[ -z "${save_freq}" ] && save_freq=100

export VLLM_ATTENTION_BACKEND=XFORMERS
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=8 \
    data.max_prompt_length=400 \
    data.max_response_length=$max_response_length \
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    +actor_rollout_ref.actor.use_trkl_loss=False \
    +actor_rollout_ref.actor.use_sqrt_trkl=False \
    +actor_rollout_ref.actor.tr_kl_loss_coef=0.0 \
    +actor_rollout_ref.actor.trpa_beta=0.0 \
    +actor_rollout_ref.actor.entropy=$entropy \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=0.7 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    +algorithm.preference_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.default_local_dir=$SAVE_DIR/$EXP_NAME \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$save_freq \
    trainer.total_epochs=12 $@ 2>&1 | tee $SAVE_DIR/$EXP_NAME.log
