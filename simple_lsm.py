import numpy as np

class LiquidStateMachine:
    def __init__(self, num_excitatory=10, num_inhibitory=20):
        """
        建立包含興奮型與抑制型神經元的 Liquid State Machine (LSM)。
        """
        self.num_exc = num_excitatory
        self.num_inh = num_inhibitory
        self.num_total = self.num_exc + self.num_inh
        
        # 初始化突觸權重矩陣 (W[i, j] 代表從神經元 j 到神經元 i 的連結)
        self.weights = np.random.randn(self.num_total, self.num_total) * 0.1
        
        # 遵循 Dale's principle：興奮型神經元的權重為正，抑制型為負
        # 前 num_exc 個神經元為興奮型
        self.weights[:, :self.num_exc] = np.abs(self.weights[:, :self.num_exc])
        # 後 num_inh 個神經元為抑制型
        self.weights[:, self.num_exc:] = -np.abs(self.weights[:, self.num_exc:])
        
        # 移除自我連結 (Self-connections)
        np.fill_diagonal(self.weights, 0)
        
        # 隨機生成稀疏連結 (假設只有 30% 的連結存在)
        sparsity_mask = np.random.rand(self.num_total, self.num_total) < 0.3
        self.weights *= sparsity_mask
        
        # 神經元狀態 (Leaky Integrate-and-Fire 模型)
        self.membrane_potentials = np.zeros(self.num_total) # 膜電位
        self.threshold = 1.0 # 激發閾值
        self.leak = 0.9 # 漏電流衰減率
        
    def step(self, input_current=0):
        """模擬一個時間步長的神經元狀態更新"""
        # 判斷哪些神經元達到閾值並產生脈衝 (Spike)
        spikes = self.membrane_potentials >= self.threshold
        
        # 將產生脈衝的神經元電位重置為 0
        self.membrane_potentials[spikes] = 0.0
        
        # 計算其他神經元傳遞過來的突觸輸入
        synaptic_input = np.dot(self.weights, spikes.astype(float))
        
        # 更新膜電位：衰減 + 突觸輸入 + 外部輸入
        self.membrane_potentials = (self.membrane_potentials * self.leak) + synaptic_input + input_current
        
        return spikes
        
    def update_connections(self, mode="prune", threshold=0.05):
        """
        動態改變突觸的連結。
        提供了一個 pruning (修剪) 的範例：移除權重過小的連結。
        """
        if mode == "prune":
            # 找到權重絕對值大於閾值的連結
            keep_mask = np.abs(self.weights) >= threshold
            old_count = np.count_nonzero(self.weights)
            
            # 修剪掉權重過小的連結
            self.weights *= keep_mask
            
            new_count = np.count_nonzero(self.weights)
            print(f"[突觸更新] 已修剪 {old_count - new_count} 個微弱的突觸連結。")
            
    def display_info(self):
        """顯示目前的網路狀態"""
        print(f"總神經元數: {self.num_total} (興奮型: {self.num_exc}, 抑制型: {self.num_inh})")
        print(f"目前活躍的突觸連結數量: {np.count_nonzero(self.weights)}")

if __name__ == "__main__":
    # 1. 建立一個包含 10 個興奮型、20 個抑制型神經元的 LSM
    lsm = LiquidStateMachine(num_excitatory=10, num_inhibitory=20)
    
    print("--- 初始狀態 ---")
    lsm.display_info()
    
    print("\n--- 開始模擬 (5 個時間步長) ---")
    # 2. 模擬給予隨機輸入的情境
    for t in range(5):
        # 隨機給予一些神經元外部電流
        random_input = np.random.rand(lsm.num_total) * 0.5
        spikes = lsm.step(input_current=random_input)
        print(f"Time Step {t+1}: 有 {np.sum(spikes)} 個神經元被激發 (Spiked)。")
        
    print("\n--- 視情況改變突觸連結 ---")
    # 3. 動態改變連結 (例如：修剪掉不重要的微弱連結)
    lsm.update_connections(mode="prune", threshold=0.03)
    lsm.display_info()
