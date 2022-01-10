for ((i=0;i<5;i=i+1))
do
	save_path="./rest/final/dpav02_dqn/$i"
	mkdir -p $save_path

	python run.py \
	--agt 12 \
	--usr 2 \
	--max_turn 40 \
	--kb_path ./deep_dialog/data_restaurant/restaurant.kb.1k.v1.p \
	--goal_file_path ./deep_dialog/data_restaurant/user_goals_first.v1.p \
	--slot_set ./deep_dialog/data_restaurant/restaurant_slots.txt \
	--act_set ./deep_dialog/data_restaurant/dia_acts.txt \
	--dict_path ./deep_dialog/data_restaurant/slot_dict.v2.p \
	--nlg_model_path ./deep_dialog/models/nlg/restaurant/lstm_tanh_[1532068150.19]_98_99_294_0.983.p \
	--nlu_model_path ./deep_dialog/models/nlu/restaurant/lstm_[1532107808.26]_68_74_20_0.997.p \
	--diaact_nl_pairs ./deep_dialog/data_restaurant/sim_dia_act_nl_pairs.v2.json \
	--dqn_hidden_size 80 \
	--experience_replay_pool_size 10000 \
	--episodes 300 \
	--simulation_epoch_size 500 \
	--write_model_dir $save_path \
	--run_mode 3 \
	--act_level 0\
	--slot_err_prob 0.00 \
	--intent_err_prob 0.00 \
	--batch_size 16 \
	--warm_start 1 \
	--warm_start_epochs 120 \
	--epsilon 0.00 \
	--gamma 0.95 \
	--evaluation True\
	--icm 0 \
	--per 0 \
	--noisy 0\
	--dpav 1\
	--dueling_dqn 0 \
	--double_dqn 0\
	--bayesian 0\
	--averaged 0\
	--maxmin 0\
	--sunrise 0\
	--discount_factor 0.2\
	--num_target_net 4\
	--distributional 0
done