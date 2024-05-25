# import torch
import datetime
from config import get_config
from runner.shared.sim_runner_ms import Runner # For torch:sim_runner, For mindspore:sim_runner_ms
# You should also change items in utils/replay_buffer between mindspore and torch




def main(args):
    runner = Runner(args)
    runner.run(60 * 10)  # 10




if __name__ == '__main__':

    start_time = datetime.datetime.now()
    parser = get_config()

    args = parser.parse_args()
    main(args)
    end_time = datetime.datetime.now()
    print("runtime: "+str(end_time-start_time))
