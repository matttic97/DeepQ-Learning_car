

class ManualDriver:
    def __init__(self, game):
        self.game = game
        
    def action_from_keys(self, keys):
        action = 0
        if keys[self.game.K_SPACE] and keys[self.game.K_UP]:
            action = 0
        elif keys[self.game.K_SPACE] and keys[self.game.K_LEFT]:
            action = 5
        elif keys[self.game.K_SPACE] and keys[self.game.K_RIGHT]:
            action = 6
        elif keys[self.game.K_UP] and keys[self.game.K_LEFT]:
            action = 3
        elif keys[self.game.K_UP] and keys[self.game.K_RIGHT]:
            action = 4
        elif keys[self.game.K_UP]:
            action = 1
        elif keys[self.game.K_SPACE]:
            action = 2
        elif keys[self.game.K_LEFT]:
            action = 7
        elif keys[self.game.K_RIGHT]:
            action = 8
        return action
    
    def next_action(self, state):
        keys = self.game.key.get_pressed()
        action = self.action_from_keys(keys)
        return action
