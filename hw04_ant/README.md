# RL Homework

### Task
В данном задании необходимо обучить агента побеждать в игре Ant при помощи метода DDPG, TD3 или SAC. Для решения задачи можно трансформировать состояние и награду среды.
К заданию также нужно приложить код обучения агента (не забудьте зафиксировать seed!), готовый (уже обученный) агент должен быть описан в классе Agent в файле `agent.py`.

### Оценка:
От 1 до 10, баллы начисляются за полученные агентом очки в среднем за 50 эпизодов. Максимальный балл соответствует `2000` очкам и выше, минимальный - `500`

# Result

Best reward under 1M steps:

`Step: 830001, Reward mean: 2229.4846282514927, Reward std: 39.55371539623023, Alpha: 0.02095917984843254`

Last:

`Step: 1260001, Reward mean: 2483.6213500917083, Reward std: 45.82071055152405, Alpha: 0.03042907826602459`

![test_run](best_agent.gif)