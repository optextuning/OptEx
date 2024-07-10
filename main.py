# from baselines.demo_tuning import DemoTuner
# from baselines.prefix_tuning import PrefixTuner
# from baselines.prompt_tuning import PromptTuner
from DifficultyEstimation.cartography import DatasetCartographyGenerativeTask

# 4 Approaches
# 1. Baseline 1: DemoTuning
# 2. Baseline 2: PrefixTuning
# 3. Baseline 3: PromptTuning
# 4. OptEx

# 5 Models
# 1. T5-3B
# 2. Tk-Instruct 3B
# 3. Llama 3 8b
# 4. Llama 3 8b instruct
# 5. Llama 3 70B

# 3 Scenarios for OptEx Tuning
# 1. Regular Instruction Tuning Vanilla Model

# 2. OptEx Instruction Tuning Vanilla Model
# 2.a OptEx Instruction Tuning Vanilla Model - 2 diff, 2 easy, 2 medium
# 2.b OptEx Instruction Tuning Vanilla Model - 1 diff, 1 easy, 1 medium
# 2.c OptEx Instruction Tuning Vanilla Model - 2 diff
# 2.d OptEx Instruction Tuning Vanilla Model - 2 med
# 2.e OptEx Instruction Tuning Vanilla Model - 2 easy

# 3. OptEx Instruction Tuning Instruction Tuned Model - Best Performing Scenario From 2
