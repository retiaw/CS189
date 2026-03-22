from CS189.utils import get_args, get_method, show_results
from CS189.lib.data import Data


if __name__ == '__main__':
    args = get_args()
    train_data, test_data, info = Data(args.data_path, args.data_name).get_data_from_TALENT()

    method = get_method(args.model_type)(args, info)
    method.fit(train_data)
    vres, metric_names, _ = method.predict(test_data)

    show_results(vres, metric_names)
