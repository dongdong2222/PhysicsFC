###########################################################
# For Game Pad
###########################################################

import pygame
import sys
import numpy as np

pygame.init()

class Joystick:
    def __init__(self, joystick_idx=4, virtual_mode=False):
        self._buttons = None
        self._axes = None
        self._hat = None
        self._joystick = None
        self._joystick_idx = joystick_idx

        self._axes = list()
        self._buttons = list()
        self._hat = list()

        # if virtual mode, it can emulate joystick.
        self._virtual_mode = virtual_mode
        if self._virtual_mode:
            self._axes = (*(np.random.rand(2) - .5) * 2, 0, *(np.random.rand(2) - .5) * 2, 0)
            self._buttons = (0,)

        # 조이스틱이 연결되어 있는지 확인
        if pygame.joystick.get_count() < 1:
            print("조이스틱을 연결해주세요.")
            # sys.exit()
        else:
            try:
                # 첫 번째 조이스틱 초기화
                self._joystick = pygame.joystick.Joystick(self._joystick_idx)
                self._joystick.init()

                # 조이스틱의 정보 출력
                print("Joystick Name:", self._joystick.get_name())
                print("Number of Buttons:", self._joystick.get_numbuttons())
                print("Number of Axes:", self._joystick.get_numaxes())
                print("Number of Hats:", self._joystick.get_numhats())

                self._buttons = list(range(self._joystick.get_numbuttons()))
                self._axes = list(range(self._joystick.get_numaxes()))
                self._hat = int()
            except:
                self._joystick = None
    
    def is_available(self):
        return self._joystick

    def is_virtual_mode(self):
        return self._virtual_mode
    
    def command(self):
        play = True
        if self.is_available():
            while play:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                # 조이스틱의 모든 버튼 상태 출력
                for i in range(self._joystick.get_numbuttons()):
                    self._buttons[i] = self._joystick.get_button(i)

                # 조이스틱의 모든 축 상태 출력
                for i in range(self._joystick.get_numaxes()):
                    self._axes[i] = self._joystick.get_axis(i)

                # 조이스틱의 모든 hat 상태 출력
                for i in range(self._joystick.get_numhats()):
                    self._hat = self._joystick.get_hat(i)

                yield self._buttons, self._axes, self._hat

        elif self._virtual_mode:
            while play:
                yield self._buttons, self._axes, self._hat
        else:
            return None, None, None
    
    def update_virtual_command(self):
        self._axes = (*(np.random.rand(2) - .5) * 2, 0, *(np.random.rand(2) - .5) * 2, 0)
        self._buttons = (0,)

# for keyboard control
def command():
    play = True
    while play:
        _command = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT: #game의 event type이 QUIT 명령이라면
                play = False # while문을 종료시킴
                pygame.quit()
            
            if event.type == pygame.KEYDOWN: #game의 event type이 키보드를 누른 상태일 때
                if event.key == pygame.K_UP:
                    _command = 'up'
                elif event.key == pygame.K_DOWN:
                    _command = 'down'
                elif event.key == pygame.K_RIGHT:
                    _command = 'right'
                elif event.key == pygame.K_LEFT:
                    _command = 'left'
                elif event.key == pygame.K_q:
                    _command = 'q'
                elif event.key == pygame.K_w:
                    _command = 'w'
                elif event.key == pygame.K_e:
                    _command = 'e'
                elif event.key == pygame.K_r:
                    _command = 'r'
                elif event.key == pygame.K_a:
                    _command = 'a'
                elif event.key == pygame.K_s:
                    _command = 's'
                elif event.key == pygame.K_d:
                    _command = 'd'
                elif event.key == pygame.K_f:
                    _command = 'f'
                
            elif event.type == pygame.KEYUP:
                _command = 'q'
        
        yield _command
