for ((i=0;i<1;i=i+1))
do
  save_path="./movie/final/dqn/$i"
  mkdir -p $save_path

  python2 run.py \
  --agt 9 \
  --usr 1 \
  --max_turn 40 \
  --movie_kb_path ./deep_dialog/data/movie_kb.1k.p \
  --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p \
  --write_model_dir $save_path \
  --dqn_hidden_size 80 \
  --experience_replay_pool_size 10000 \
  --episodes 120 \
  --simulation_epoch_size 5 \
  --run_mode 3 \
  --act_level 0 \
  --slot_err_prob 0.00 \
  --intent_err_prob 0.00 \
  --batch_size 16 \
  --warm_start 1 \
  --warm_start_epochs 120 \
  --planning_steps 4 \
  --boosted 0 \
  --train_world_model 0 \
  --num_target_net 4\
  --model_type DQN

done