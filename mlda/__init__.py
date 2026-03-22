#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )

import serket as srk
import numpy as np
import os

from . import mlda
from ._common import as_modality_array

class MLDA(srk.Module):
    def __init__(
        self,
        K,
        weights=None,
        itr=100,
        name="mlda",
        category=None,
        load_dir=None,
        backend="auto",
        num_restarts=None,
        random_state=None,
    ):
        super(MLDA, self).__init__(name, True)
        self.__K = K
        self.__weights = weights
        self.__itr = itr
        self.__category = category
        self.__load_dir = load_dir
        self.__backend = backend
        self.__num_restarts = num_restarts
        self.__random_state = random_state
        self.__n = 0
        
    def update(self, load_trained_model=None):
        data = self.get_observations()
        Pdz = self.get_backward_msg() # P(z|d)

        M = len( data )     # モダリティ数
        
        for m in range(M):
            if as_modality_array(data[m]) is not None:
                N = len( data[m] )     # データ数
                break
        
        # backward messageがまだ計算されていないときは一様分布にする
        if Pdz is None:
            Pdz = np.ones( (N, self.__K) ) / self.__K
    
        # データの正規化処理
        for m in range(M):
            if as_modality_array(data[m]) is not None:
                data[m][ data[m]<0 ] = 0
            
        if self.__weights is not None:
            for m in range(M):
                if as_modality_array(data[m]) is not None:
                    divider = np.where( data[m].sum(1)==0, 1, data[m].sum(1) )
                    data[m] = ( data[m].T / divider ).T * self.__weights[m]
        
        for m in range(M):
            if as_modality_array(data[m]) is not None:
                data[m] = np.array( data[m], dtype=np.int32 )

        model_dir = self.__load_dir if load_trained_model is None else load_trained_model
        if model_dir is None:
            save_dir = os.path.join( self.get_name(), "%03d" % self.__n )
        else:
            save_dir = os.path.join( self.get_name(), "%03drecog" % self.__n )
        
        # MLDA学習
        seed = None if self.__random_state is None else self.__random_state + self.__n
        Pdz, Pdw = mlda.train(
            data,
            self.__K,
            self.__itr,
            save_dir,
            Pdz,
            self.__category,
            model_dir,
            backend=self.__backend,
            num_restarts=self.__num_restarts,
            random_state=seed,
        )
        
        self.__n += 1
        
        # メッセージの送信
        self.set_forward_msg( Pdz )
        self.send_backward_msgs( Pdw )
