import joblib
import json
from pprint import pprint

def main():
    path = 'enhanced_svm_model.pkl'
    obj = joblib.load(path)
    print('TYPE:', type(obj).__name__)
    if isinstance(obj, dict):
        print('TOP_LEVEL_KEYS:', list(obj.keys()))
        for k, v in obj.items():
            print('KEY:', k, '=>', type(v).__name__)
            if isinstance(v, dict):
                print('  SUBKEYS:', list(v.keys()))
            elif hasattr(v, 'get_params'):
                print('  MODEL_PARAMS_KEYS:', list(v.get_params().keys())[:10])
    else:
        print('DIR_ATTRS:', [a for a in dir(obj) if not a.startswith('_')][:30])

if __name__ == '__main__':
    main()

