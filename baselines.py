import numpy as np

class FixedPolicy:
    """Fixed (s,S) Policy: 不主动调整库存规则，不加急，采用默认配给"""
    def predict(self, obs, mask=None):
        return 0  # 对应 l_t=0, e_t=0, g_t=0


class RuleBasedPolicy:
    """启发式 Rule-based 规则策略"""
    def predict(self, obs, mask=None):
        # 从 Observation 解码中断状态
        a_t = int(round(obs[-3] * 2))
        u_t = int(round(obs[-2] * 2))
        
        if u_t == 2:  # 完全运输中断
            l_t = 2  # 建立最高库存缓冲
            e_t = 0  # 禁加急
            g_t = 1  # 优先低 DOS 保证防断供
        elif u_t == 1 or a_t >= 1:  # 轻度延误或产能受限
            l_t = 1
            e_t = 1  # 开启加急
            g_t = 1
        else:  # 正常运营
            l_t = 0
            e_t = 0
            g_t = 0
            
        action_idx = l_t * 6 + e_t * 3 + g_t
        return action_idx