python run.py --source 1 --target 2 -b 0.2 0.4 0.6 0.8 1.0 -r 10 --gpu 0 --config ../hyperopt/best_configs/transfer_hyperopt_one\&two@0.80.json
python run.py --source 1 --target 3 -b 0.2 0.4 0.6 0.8 1.0 -r 10 --gpu 0 --config ../hyperopt/best_configs/transfer_hyperopt_one\&three@0.80.json
python run.py --source 1 --target 4 -b 0.2 0.4 0.6 0.8 1.0 -r 10 --gpu 0 --config ../hyperopt/best_configs/transfer_hyperopt_one\&four@0.80.json

python run.py --source 2 --target 1 -b 0.2 0.4 0.6 0.8 1.0 -r 10 --gpu 0 --config ../hyperopt/best_configs/transfer_hyperopt_two\&one@0.80.json
python run.py --source 2 --target 3 -b 0.2 0.4 0.6 0.8 1.0 -r 10 --gpu 0 --config ../hyperopt/best_configs/transfer_hyperopt_two\&three@0.80.json
python run.py --source 2 --target 4 -b 0.2 0.4 0.6 0.8 1.0 -r 10 --gpu 0 --config ../hyperopt/best_configs/transfer_hyperopt_two\&four@0.80.json