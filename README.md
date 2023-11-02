# Jockerding-flask

클라이언트 단에서 
imageData = captureCanvas.toDataURL('image/png');
와 같이 이미지 데이터를 얻고,
서버로 전송

서버에서는
base64Data = imageData.replace(/^data:image\/png;base64,/, '');          
          try {
            const response = await axios.post('http://localhost:5000/predict', {
              base64Data: base64Data,
            });

이런식으로 5000번 포트로 base64 data를 predict 서버로 전송하면

이미지에 대한 응답을 서버로 보내줌.
