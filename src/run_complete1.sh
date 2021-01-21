python run_complete.py --source 1 --target 2 -b 0.2 0.4 0.6 0.8 1.0 -p 10 --best_only --gpu 0 --arch_config ../hyperopt/best_configs/transfer_hyperopt_one\&two@0.80.json --pre_config ../hyperopt/best_configs/transfer_pre_hyperopt_one\&two@0.80.json
python run_complete.py --source 1 --target 3 -b 0.2 0.4 0.6 0.8 1.0 -p 10 --best_only --gpu 0 --arch_config ../hyperopt/best_configs/transfer_hyperopt_one\&three@0.80.json --pre_config ../hyperopt/best_configs/transfer_pre_hyperopt_one\&three@0.80.json
python run_complete.py --source 1 --target 4 -b 0.2 0.4 0.6 0.8 1.0 -p 10 --best_only --gpu 0 --arch_config ../hyperopt/best_configs/transfer_hyperopt_one\&four@0.80.json --pre_config ../hyperopt/best_configs/transfer_pre_hyperopt_one\&four@0.80.json

python run_complete.py --source 2 --target 1 -b 0.2 0.4 0.6 0.8 1.0 -p 10 --best_only --gpu 0 --arch_config ../hyperopt/best_configs/transfer_hyperopt_two\&one@0.80.json --pre_config ../hyperopt/best_configs/transfer_pre_hyperopt_two\&one@0.80.json
python run_complete.py --source 2 --target 3 -b 0.2 0.4 0.6 0.8 1.0 -p 10 --best_only --gpu 0 --arch_config ../hyperopt/best_configs/transfer_hyperopt_two\&three@0.80.json --pre_config ../hyperopt/best_configs/transfer_pre_hyperopt_two\&three@0.80.json
python run_complete.py --source 2 --target 4 -b 0.2 0.4 0.6 0.8 1.0 -p 10 --best_only --gpu 0 --arch_config ../hyperopt/best_configs/transfer_hyperopt_two\&four@0.80.json --pre_config ../hyperopt/best_configs/transfer_pre_hyperopt_two\&four@0.80.json

