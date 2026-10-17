Overview
Real intelligence isn’t about memorizing answers - it’s knowing what to do when the problem changes.

Build systems that learn and adapt to novel, human-solvable tasks they’ve never seen before and advance AI’s ability to learn new skills efficiently.

Description
Note: This is a relaunch of ARC Prize 2025. If you are joining either this or ARC-AGI-3, please consider joining the paper track, where you can document your approach for either one of the prediction competitions.

Today’s AI systems excel at what they were trained to do, but often fall short when something unfamiliar comes along. Most benchmarks reward pattern recognition, not genuine problem-solving.

The ARC Prize focuses on true generalization: whether a system can quickly learn new skills in unfamiliar situations. Instead of rewarding pattern recognition on known tasks, it evaluates how well systems can adapt to new problems they’ve never encountered before. The evaluation environment is designed so systems can't just memorize solutions. Tasks take place in hidden, interactive environments that require exploration and multi-step reasoning.

Your solution could help move AI closer to systems that learn the way people do: flexible, efficient, and able to handle new challenges.

The real test of intelligence begins when the problem changes.

Evaluation
This competition evaluates submissions on the percentage of correct predictions. For each task, you should make 2 attempts to predict the exact outputs for every test input grid contained in the task. (Tasks can have more than one test input that needs a predicted output.) Each task test output has one ground truth. For a given task output, any of the 2 predicted outputs matches the ground truth exactly, you score 1 for that task test output, otherwise 0. The final score is the sum averaged of the highest score per task output divided by the total number of task test outputs.

Submission File
The submission file for this competition must be a json named submission.json.

For each task output in the evaluation set, you should make exactly 2 predictions (attempt_1, attempt_2). The structure of predictions is shown below. Many tasks have multiple outputs (a multiple dictionaries enclosed in a list), although some tasks have a single output that must be predicted. When a task has multiple test outputs that need to be predicted (e.g., task 12997ef3 below), they must be in the same order as the corresponding test inputs.

IMPORTANT: All the task_ids in the input challenges json file must also be present in the submission.json file. Both "attempt_1" and "attempt_2" must be present, even if your submission doesn't have 2 predictions.

{"00576224": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
 "009d5c81": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
 "12997ef3": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]},
              {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
 ...
}

Timeline
March 25, 2026 - Start Date.
October 26, 2026 - Entry Deadline. You must accept the competition rules before this date in order to compete.
October 26, 2026 - Team Merger Deadline. This is the last day participants may join or merge teams.
November 2, 2026 - Final Submission Deadline.
December 4, 2026 - Winners announcement.
All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

Prizes
TOTAL PRIZES AVAILABLE: $700,000
Progress Prizes: $275,000
Grand Prize: $275,000
Bonus Prize: $150,000
In line with the spirit of the competition, participants eligible for a prize will be removed from the competition if they do not open source their solutions.

ARC-AGI-2 Progress Prizes: $275,000
First Prize: $75,000
Second Prize: $50,000
Third Prize: $40,000
Fourth Prize: $35,000
Fifth Prize: $25,000
Sixth Prize: $20,000
Seventh Prize: $15,000
Eighth Prize: $15,000
ARC-AGI-2 Grand Prize: $275,000
The Grand Prize will be awarded to the highest scoring Solution Writeup based on the below criteria. All artifacts should be open sourced and attached to an official competition Solution Writeup within seven days of the competition's submission deadline to be considered eligible.

Submissions for the Grand Prize are evaluated equally across the following six criteria. Each criterion is scored on a scale from 0 (lowest) to 5 (highest), with the final score calculated as the average of all six.

Category   	Description	 
Accuracy	How accurate is the submission based on its performance on the leaderboard? 	 
Universality 	How general and universal is the Submission approach beyond the competition? Does your submission translate to other similar problems? How well does your method generalize?	
Progress	How much does your Solution increase the overall chance of anyone achieving 85% on ARC-AGI-2?	
Theory	How well do the artifacts describe why the Submission works (as opposed to merely describing how it works)?	
Completeness	How thoroughly and completely does the solution cover your submission to the leaderboard?	
Novelty	How novel is the Submission relative to existing public research?	
Bonus Prize
A Bonus Prize of an additional $150,000 will be unlocked in the event that a team achieves a score of at least 85% accuracy on the competition leaderboard. At the end of the competition, the Bonus Prize will be divided among the Top 5 teams that have achieved 85% accuracy as outlined below. In the event that fewer than 5 teams have achieved 85% accuracy, those prizes will be divided proportionately among qualifying teams.

First Prize: $75,000
Second Prize: $25,000
Third Prize: $20,000
Fourth Prize: $20,000
Fifth Prize: $10,000

Code Requirements

This is a Code Competition
NOTE: We are currently exploring options to provide better compute for this competition.

Submissions to this competition must be made through Notebooks. In order for the "Submit to Competition" button to be active after a commit, the following conditions must be met:

CPU Notebook <= 12 hours run-time
GPU Notebook <= 12 hours run-time
No internet access enabled
External data, freely & publicly available, is allowed, including pre-trained models
Submission file must be named submission.json
Submission runtimes have been obfuscated. If you repeat the exact same submission you will see up to 10 minutes of variance in the time before you receive your score.

Please see the Code Competition FAQ for more information on how to submit.

Upgraded Accelerators
This competition has access to Kaggle's pool of powerful new L4x4 machines! These machines offer 96GB of GPU memory enabling submissions with much larger models. See this page for more on these machines.

What you need to know:

Quota usage - Due to their limited availability, notebooks with L4x4s consume GPU quota at twice the rate of the older T4x2 and P100 machines. We may increase this rate as necessary to ensure these machines are available for submission scoring.
Restricted Use - L4s are only available for notebooks attached to this competition. We will build tooling to enforce this if necessary. In the meantime, attempts to circumvent this restriction may be enforced by Kaggle moderation with consequences including team bans from the competition or account bans.
No Internet - All L4 sessions must have internet disabled.

Citation
Francois Chollet, Mike Knoop, Greg Kamradt, Walter Reade, María Cruz, and Addison Howard. ARC Prize 2026 - ARC-AGI-2. https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2, 2026. Kaggle.

