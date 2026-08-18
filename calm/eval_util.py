
import numpy as np
import os
from attrdict import AttrMap

import matplotlib.pyplot as plt
from scipy.stats import norm, expon, lognorm

# EvaluationData - task/metric 별 데이터 저장 및 분석을 위한 클래스
# gain_time으로 끝날 경우 성공과 관련된 metric
# 따라서 mean, std에 더해 summary에 rate가 추가된다.
# gain time의 경우 0미만의 값이면 성공하지 못한 것, 이상의 경우 성공한 것으로 계산한다. 시간 단위는 frame.
class EvaluationData():
    task_metric_dict = {
        # "move": [
        #     'Character Moving Direction Deviation',
        #     'Time to achieve Character Moving Direction',
        #     'Character Facing Direction Deviation',
        #     'Time to achieve Character Facing Direction',
        #     'Character Speed Deviation',
        #     'Time to achieve Character Speed',
        # ],
        "move": [
            'char_tgt_move_diff',
            'char_tgt_move_gain_time',
            'char_tgt_forward_diff',
            'char_tgt_forward_gain_time',
            'char_tgt_speed_diff',
            'char_tgt_speed_gain_time'
        ],
        "dribble": [
            'Character-Ball Distance',
            'Foot-Ball Distance'
            'Time to achieve Ball Moving Direction',
            'Time to achieve Ball Speed',
            'Ball Moving Direction Deviation',
            'Ball Speed Deviation',
            'Ball Acquisition Time',
        ],
        "dribble2-1": [
            'Ball Speed',
            'Character-Ball Distance',
            'Foot-Ball Distance'
        ],
        "dribble2-2": [
            'Ball Speed',
            'Character-Ball Distance',
            'Foot-Ball Distance'
        ],
        "dribble2-3": [
            'Ball Speed',
            'Character-Ball Distance',
            'Foot-Ball Distance'
        ],
        "dribble2-4": [
            'Ball Speed',
            'Character-Ball Distance',
            'Foot-Ball Distance'
        ],
        "dribble2-5": [
            'Ball Speed',
            'Character-Ball Distance',
            'Foot-Ball Distance'
        ],
        "dribble2-6": [
            'Ball Speed',
            'Character-Ball Distance',
            'Foot-Ball Distance'
        ],
        "dribble2-7": [
            'Ball Speed',
            'Character-Ball Distance',
            'Foot-Ball Distance'
        ],
    }
    
    def __init__(self) -> None:
        self.data = AttrMap(sequence_type=np.ndarray)
        self.data.update({
            task: {metric : np.array([], dtype=float) for metric in self.task_metric_dict[task]} for task in self.task_metric_dict.keys()
        })
        self.summary = AttrMap(sequence_type=np.ndarray)
        self.summary.update({
            task: {metric: {
                "mean": np.array([], dtype=float),
                "std": np.array([], dtype=float),
                "lognorm_mean": np.array([], dtype=float),
                "lognorm_var": np.array([], dtype=float),
                **({"rate": np.array([], dtype=float)} if metric.startswith("Time to achieve") else {})
            } for metric in self.task_metric_dict[task]} for task in self.task_metric_dict.keys()
        })
    
    def update_data(self, metric:str, new_data:np.ndarray, task:str="move") -> None:
        self.data[task][metric] = np.concatenate([self.data[task][metric], np.array(new_data, dtype=float)], axis=-1)
    
    def refresh_summary(self):
        updated = 0
        for task in self.task_metric_dict.keys():
            for metric in self.task_metric_dict[task]:
                if len(self.data[task][metric]) != 0:
                    updated+=1
                    if metric.startswith("Time to achieve"):
                        success_idxs = np.nonzero(np.greater_equal(self.data[task][metric], np.zeros_like(self.data[task][metric])))
                        success_count = np.count_nonzero(np.greater_equal(self.data[task][metric], np.zeros_like(self.data[task][metric])))
                        if success_count > 0:
                            self.summary[task][metric]["mean"] = np.mean(self.data[task][metric][success_idxs], axis=-1)
                            self.summary[task][metric]["std"] = np.std(self.data[task][metric][success_idxs], axis=-1)
                            self.summary[task][metric]["rate"] = success_count / np.size(self.data[task][metric])
                        else:
                            self.summary[task][metric]["mean"] = np.array([0.])
                            self.summary[task][metric]["std"] = np.array([0.])
                            self.summary[task][metric]["lognorm_mean"], self.summary[task][metric]["lognorm_var"] = np.array([0.]), np.array([0.])
                            self.summary[task][metric]["rate"] = np.array([0.])
                    else:
                        self.summary[task][metric]["mean"] = np.mean(self.data[task][metric], axis=-1)
                        self.summary[task][metric]["std"] = np.std(self.data[task][metric], axis=-1)
                        
        return updated
    
    def save_csv(self, path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/summary/eval_result.csv")) -> None:
        import csv
        # # 헤더 작성
        # headers = ["task"]
        # for metric in self.task_metric_dict["move"]:
        #     headers.append(f"{metric}_mean")
        #     headers.append(f"{metric}_std")
        #     if metric.startswith("Time to achieve"):
        #         headers.append(f"{metric}_rate")
        
        # CSV 데이터 구성
        rows = []
        for task in self.task_metric_dict.keys():
            header_row = [""]
            row = [task]
            for metric in self.task_metric_dict[task]:
                header_row.append(f"{metric}_mean")
                header_row.append(f"{metric}_std")
                row.append(self.summary[task][metric]["mean"])
                row.append(self.summary[task][metric]["std"])
                if metric.startswith("Time to achieve"):
                    header_row.append(f"{metric}_rate")
                    row.append(self.summary[task][metric]["rate"])
            rows.append(header_row)
            rows.append(row)
        
        # CSV 파일로 저장
        with open(path, mode="w", newline="") as file:
            writer = csv.writer(file)
            # writer.writerow(headers)  # 헤더 작성
            writer.writerows(rows)    # 데이터 작성
    
    def save_npz(self, path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/summary/eval_result.npz")) -> None:
        # Data dict를 npz로 저장
        np.savez(path, **{
            f"data_{task}_{metric}": self.data[task][metric]
            for task in self.task_metric_dict.keys()
            for metric in self.task_metric_dict[task]
        })
        
        # print(f"data size: {self.data.dribble.char_tgt_move_diff[np.argwhere(np.isnan(self.data.move.char_tgt_move_diff))].shape}")

    def load_npz(self, path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/summary/eval_result.npz"), override_task=None) -> None:
        # .npz 파일에서 데이터 로드 후 update_data 호출
        data = np.load(path)
        for task in self.task_metric_dict.keys():
            if task != override_task:
                for metric in self.task_metric_dict[task]:
                    if f"data_{task}_{metric}" in data:
                        self.update_data(metric, data[f"data_{task}_{metric}"], task=task)
                        print(f"{task}:{metric} - numdata: {data[f'data_{task}_{metric}'].shape}")
        # Refresh summary
        self.refresh_summary()
        
    def save_graphs(self, path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/summary/graphs")):
        if not os.path.exists(path):
            os.makedirs(path)

        for task in self.task_metric_dict.keys():
            for metric in self.task_metric_dict[task]:
                if len(self.data[task][metric]) == 0:
                    continue
                
                data = self.data[task][metric]
                if metric.startswith("Time to achieve"):
                    data=data[data>=0]
                
                print(f"{task} {metric} data nan at: {np.argwhere(np.isnan(self.data[task][metric]))}")
                # print(self.data[task][metric])
                
                # 데이터의 실제 분포
                plt.figure(figsize=(10, 6))
                plt.hist(data.flatten(), bins=20, density=True, alpha=0.6, color="blue", label="Actual Data")

                # 정규분포 그래프
                mean, std = np.mean(data), np.std(data)
                x = np.linspace(min(data), max(data), 1000)
                normal_pdf = norm.pdf(x, mean, std)
                plt.plot(x, normal_pdf, 'g--', lw=2, label="Normal Distribution Fit")

                if metric.startswith("Time to achieve"):
                    # 로그 정규분포 그래프
                    mean_ln, var_ln, mean_log, var_log = log_normal_stats(data)
                    lognorm_pdf = lognorm.pdf(x, s=np.sqrt(var_log), scale=np.exp(mean_log))
                    plt.plot(x, lognorm_pdf, 'r-', lw=2, label="Log-normal Distribution Fit")
                else:
                    # 지수분포 그래프
                    lambda_param = 1 / np.mean(data)  # 지수분포의 파라미터 계산
                    exp_pdf = expon.pdf(x, scale=1/lambda_param)
                    plt.plot(x, exp_pdf, 'r-', lw=2, label="Exponential Distribution Fit")

                # 그래프 설정
                plt.title(f"{task} - {metric}")
                plt.xlabel("Value")
                plt.ylabel("Density")
                plt.legend()

                # 그래프 저장
                graph_filename = os.path.join(path, f"{task}_{metric}.png")
                plt.savefig(graph_filename)
                plt.close()
                

def log_normal_stats(data):
    log_data = np.log(data)
    
    # 로그 평균 및 분산 계산
    mean_log = np.mean(log_data)
    var_log = np.var(log_data, ddof=0)
    
    # 큰 값으로 인한 오버플로우 방지를 위해 클리핑
    max_exp_input = 50
    var_log = np.clip(var_log, None, max_exp_input)
    mean_log = np.clip(mean_log, None, max_exp_input)

    mean_ln = np.exp(mean_log + 0.5 * var_log)
    var_ln = (np.exp(var_log) - 1) * np.exp(2 * mean_log + var_log)
    
    return mean_ln, var_ln, mean_log, var_log

def get_default_summary_dir():
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/summary")

def get_test_cases(
    batch_num=1000, 
    batch_size=30,
    # timing_distribution="uniform", #uniform, normal
    # timing_criteria=[100, 200], #for uniform, [min, max], and for normal, [mean, std]
    save_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/summary/move_testcase.npz"), 
    override=False
) -> dict:
    
    
    data = {}

    # 목표 속력 (batch_num, batch_size): 0에서 1 사이의 스칼라 값을 생성
    target_speeds = np.random.rand(batch_num, batch_size)
    
    # 목표 이동 방향 (batch_num, batch_size, 2): 2차원 유닛 벡터 생성
    target_move_directions = np.random.randn(batch_num, batch_size, 2)
    target_move_directions /= np.linalg.norm(target_move_directions, axis=-1, keepdims=True)
    
    # 목표 facing 방향 (batch_num, batch_size, 2): 2차원 유닛 벡터 생성
    target_facing_directions = np.random.randn(batch_num, batch_size, 2)
    target_facing_directions /= np.linalg.norm(target_facing_directions, axis=-1, keepdims=True)
    
    
    data.update({
        "target_speeds": target_speeds,
        "target_move_directions": target_move_directions,
        "target_facing_directions": target_facing_directions
    })
        
    # 이미 파일이 존재하고 override가 False인 경우, 기존 파일을 불러와 반환
    if os.path.exists(save_path) and not override:
        print(f"File already exists at {save_path}. Loading existing file.")
        data_load = np.load(save_path)
        if data_load["batch_num"] != batch_num or data_load["batch_size"] != batch_size:
            print("loaded data's batch size & batch num is different! generate new data and override.")
        else:
            data.update({
                "target_speeds": data_load["target_speeds"],
                "target_move_directions": data_load["target_move_directions"],
                "target_facing_directions": data_load["target_facing_directions"]
            })
    elif os.path.exists(save_path) and override:
        print(f"File already exists at {save_path}. Overriding existing file.")
    
    # 저장 경로가 존재하지 않으면 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 데이터를 npz 형식으로 저장
    np.savez(save_path, 
             batch_num=batch_num,
             batch_size=batch_size,
             target_speeds=target_speeds, 
             target_move_directions=target_move_directions, 
             target_facing_directions=target_facing_directions)
    
    print(f"Test cases saved to {save_path}")
    
    return data


#############################################################################33
## test code
## to test implementation of eval_util, just run "python eval_util.py"
if __name__ == '__main__':
    
    result = EvaluationData()
    result.load_npz("/home/nogabi/Data1/nogabi/soccer/football_01/workspace/CALM/calm/data/summary/eval_cs150_bn100_bs30_angle10_speed0.5_distance0.5.npz")
    result.save_graphs()
    
    # print("\nTest Start!\n")
    # test_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"data/summary/test_eval_util.npz")
    # test_csv = os.path.join(os.path.abspath(os.path.dirname(__file__)),"data/summary/test_eval_util.csv")
    # test_save_load_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"data/summary/test_save_load.npz")
    # print("Testing generation of testcase on:", test_path, ",", test_csv, ",", test_save_load_path)
    # print("Be sure that the testing file path is different from the evaluation test case!")
    
    # print("\nAre you SURE to process? (y/n):",end=" ")
    
    # answer = input()
    # if answer.lower() != 'y':
    #     print("Stop Testing......")
    #     exit()
    
    # # generate new test cases
    # get_test_cases(save_path=test_path, override=True)
    # testcase = get_test_cases(save_path=test_path)
    # for key in testcase.keys():
    #     if key != "target_speeds":
    #         np.testing.assert_almost_equal(np.linalg.norm(testcase[key],axis=-1), np.ones_like(testcase[key][..., 0]), err_msg=f"{key} not unit!")
    # print("testcase assertion passed!")
    
    # e = EvaluationData()
    # print("\nEvaluationData.data:\t",e.data)
    # print("\nEvaluationData.summary:\t",e.summary)
    # print("\nmove.char_tgt_move_dir:\t", e.data.move.char_tgt_move_dir)
    # e.update_data(
    #     metric='char_tgt_move_dir',
    #     new_data=np.random.rand(4),
    # )
    
    # e.update_data(
    #     metric='char_tgt_move_dir_gain_time',
    #     new_data=np.random.rand(10) - 0.5,
    # )
    
    # print("\nEvaluationData.data:\t",e.data)
    # print("\nEvaluationData.summary:\t",e.summary)
    # print("\n\nrefreshed metrics: ", e.refresh_summary(), "\n")
    # print("\nEvaluationData.summary:\t",e.summary)
    # # print("\nmove.char_tgt_move_dir.mean:\t",e.summary.move.char_tgt_move_dir.mean)
    
    # e.save_csv(path=test_csv)
    
    # os.remove(test_path)
    
    # print(f"Test file [{test_path}] removed!")
    
    # e.save_npz(test_save_load_path)
    
    # ee = EvaluationData()
    # ee.load_npz(test_save_load_path)
    
    # np.testing.assert_almost_equal(e.summary.move.char_tgt_move_dir.mean, ee.summary.move.char_tgt_move_dir.mean, err_msg="loaded mean is not equal!")
    # np.testing.assert_almost_equal(e.summary.move.char_tgt_move_dir.std, ee.summary.move.char_tgt_move_dir.std, err_msg="loaded mean is not equal!")
    # print("save & load test passed!")
    
    # os.remove(test_save_load_path)
    # print(f"Test file [{test_save_load_path}] removed!")
    
    