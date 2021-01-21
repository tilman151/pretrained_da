
python run_complete.py --source 3 --target 1 -b 0.2 0.4 0.6 0.8 1.0 -p 10 --best_only --gpu 1 --arch_config ../hyperopt/best_configs/transfer_hyperopt_three\&one@0.80.json --pre_config ../hyperopt/best_configs/transfer_pre_hyperopt_three\&one@0.80.json
python run_complete.py --source 3 --target 2 -b 0.2 0.4 0.6 0.8 1.0 -p 10 --best_only --gpu 1 --arch_config ../hyperopt/best_configs/transfer_hyperopt_three\&two@0.80.json --pre_config ../hyperopt/best_configs/transfer_pre_hyperopt_three\&two@0.80.json
python run_complete.py --source 3 --target 4 -b 0.2 0.4 0.6 0.8 1.0 -p 10 --best_only --gpu 1 --arch_config ../hyperopt/best_configs/transfer_hyperopt_three\&four@0.80.json --pre_config ../hyperopt/best_configs/transfer_pre_hyperopt_three\&four@0.80.json

python run_complete.py --source 4 --target 1 -b 0.2 0.4 0.6 0.8 1.0 -p 10 --best_only --gpu 1 --arch_config ../hyperopt/best_configs/transfer_hyperopt_four\&one@0.80.json  --pre_config ../hyperopt/best_configs/transfer_pre_hyperopt_four\&one@0.80.json
python run_complete.py --source 4 --target 2 -b 0.2 0.4 0.6 0.8 1.0 -p 10 --best_only --gpu 1 --arch_config ../hyperopt/best_configs/transfer_hyperopt_four\&two@0.80.json  --pre_config ../hyperopt/best_configs/transfer_pre_hyperopt_four\&two@0.80.json
python run_complete.py --source 4 --target 3 -b 0.2 0.4 0.6 0.8 1.0 -p 10 --best_only --gpu 1 --arch_config ../hyperopt/best_configs/transfer_hyperopt_four\&three@0.80.json  --pre_config ../hyperopt/best_configs/transfer_pre_hyperopt_four\&three@0.80.json
