import numpy as np
import utils

class Agent():
    STATE_DIM = 3
    TRADING_CHARGE = 0.00015    # 수수료
    TRADING_TAX = 0.0025        # 거래세

    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2
    ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]
    NUM_ACTIONS = len(ACTIONS)
    def __init__(self, environment, initial_balance, min_trading_price, max_trading_price):
        self.environment = environment
        self.initial_balance = initial_balance
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price
        
        # class attibutes
        self.balance = initial_balance  # 현재 현금 잔고
        self.num_stocks = 0             # 보유 주식 수
        # portfolio value = balance + num_stocks *(current stock value)
        self.portfolio_value = 0.
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        
        # class status
        self.ratio_hold = 0.
        self.profitloss = 0.
        self.avg_buy_price = 0.
        
    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.ratio_hold = 0.
        self.profitloss = 0.
        self.avg_buy_price = 0. 
        
    def set_balance(self, balance):
        self.initial_balance = balance
    
    def get_states(self):
        self.ratio_hold = self.num_stocks * self.environment.get_price() / self.portfolio_value
        return (self.ratio_hold,
                self.profitloss,
                (self.environment.get_price() / self.avg_buy_price) -1 if self.avg_buy_price > 0 else 0.
                )
    
    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0.
        
        pred = pred_policy
        if pred is None:
            epsilon = 1.
        else:
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1.
        
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)
        else:
            exploration = False
            action = np.argmax(pred)
    
        confidence = 0.5
        if pred_policy is not None:
            confidence = pred(action)
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])
        
        return (action, confidence, exploration)
    
    def validate_action(self, action):
        if action == Agent.ACTION_BUY:
            if self.balance < self.environment.get_price() * (1+self.TRADING_CHARGE):
                return False
        elif action == Agent.ACTION_SELL:
            if not self.num_stocks > 0:
                return False
        return True
    
    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_price
        added_training_price = max(min(
                int(confidence*(self.max_trading_price - self.min_trading_price)),
                self.max_trading_price-self.min_trading_price),
                0.)
        trading_price = self.min_trading_price + added_training_price
        return max(int(trading_price / self.environment.get_price()), 1)
    
    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD
        
        current_price = self.environment.get_price()
        if action == Agent.ACTION_BUY:
            # 매수할 단위 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = self.balance - current_price*(1+self.TRADING_CHARGE)*trading_unit
            
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0.:
                trading_unit = min(
                        int(self.balance / (current_price*(1+self.TRADING_CHARGE))),
                        int(self.max_trading_price / current_price)
                )
            
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = current_price * (1 + self.TRADING_CHARGE) * trading_unit
            if invest_amount > 0:
                self.avg_buy_price = (self.avg_buy_price * self.num_stocks + current_price * trading_unit) / (self.num_stocks + trading_unit)
                self.balance -= invest_amount
                self.num_stocks += trading_unit
                self.num_buy += 1
            
        # 매도
        elif action == Agent.ACTION_SELL:
            trading_unit = self.decide_trading_unit(confidence)     # 매도할 단위를 판단
            trading_unit = min(trading_unit, self.num_stocks)       # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            
            invest_amount = (current_price * (1 - self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            if invest_amount > 0:
                self.avg_buy_price = (self.avg_buy_price* self.num_stocks - current_price) / (self.num_stocks - trading_unit) if self.num_stocks > trading_unit else 0
            self.num_stocks -= trading_unit
            self.balance += invest_amount
            self.num_sell += 1
        
        # 관망
        else:       # Agent.ACTION_HOLD
            self.num_hold += 1
        
        # update portfolio value
        self.portfolio_value = self.balance + current_price * self.num_stocks
        self.profitloss = self.portfolio_value / self.initial_balance - 1.
        return self.profitloss
            
            
            
            
            
            
            
            
            
            
            
            