import random
from ocatari.core import OCAtari
from ocatari.utils import load_agent, parser

GAME_NAME = "Pong"
MODE = "revised"
HUD = False
env = OCAtari(GAME_NAME, mode=MODE, hud=HUD, obs_mode='dqn')
observation, info = env.reset()

opts = parser.parse_args()

if opts.path:
    agent = load_agent(opts, env.action_space.n)

for i in range(10000):
    if opts.path is not None:
        action = agent.draw_action(env.dqn_obs)
    else:
        action = random.randint(0, 0)
    obs, reward, terminated, truncated, info = env.step(action)
    # if i % 10 == 0:
    #     print(env.objects)
    #     for obj in env.objects:
    #         x, y = obj.xy
    #         if x < 160 and y < 210:
    #             opos = obj.xywh
    #             ocol = obj.rgb
    #             sur_col = make_darker(ocol)
    #             mark_bb(obs, opos, color=sur_col)
    # env.render()

    if terminated or truncated:
        print('episode finished', i)
        observation, info = env.reset()
env.close()
