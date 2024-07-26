<script lang="ts">
	const tf = window.tf;
	const trainData = {
		sizeMB: [
			0.08, 9.0, 0.001, 0.1, 8.0, 5.0, 0.1, 6.0, 0.05, 0.5, 0.002, 2.0, 0.005, 10.0, 0.01, 7.0, 6.0,
			5.0, 1.0, 1.0
		],
		timeSec: [
			0.135, 0.739, 0.067, 0.126, 0.646, 0.435, 0.069, 0.497, 0.068, 0.116, 0.07, 0.289, 0.076,
			0.744, 0.083, 0.56, 0.48, 0.399, 0.153, 0.149
		]
	};
	const testData = {
		sizeMB: [
			5.0, 0.2, 0.001, 9.0, 0.002, 0.02, 0.008, 4.0, 0.001, 1.0, 0.005, 0.08, 0.8, 0.2, 0.05, 7.0,
			0.005, 0.002, 8.0, 0.008
		],
		timeSec: [
			0.425, 0.098, 0.052, 0.686, 0.066, 0.078, 0.07, 0.375, 0.058, 0.136, 0.052, 0.063, 0.183,
			0.087, 0.066, 0.558, 0.066, 0.068, 0.61, 0.057
		]
	};
	console.log(testData.timeSec);
	console.log(testData.sizeMB);
	const trainTensor = {
		sizeMB: tf.tensor2d(trainData.sizeMB, [20, 1]),
		timeSec: tf.tensor2d(trainData.timeSec, [20, 1])
	};
	const testTensor = {
		sizeMB: tf.tensor2d(testData.sizeMB, [20, 1]),
		timeSec: tf.tensor2d(testData.timeSec, [20, 1])
	};

	const model = tf.sequential(); //레이어를 여러층으로 해서 모델을 정의하겠다.
	model.add(tf.layers.dense({ inputShape: [1], units: 1 })); //신경망 밀집층
	model.compile({ optimizer: 'sgd', loss: 'meanAbsoluteError' });
	console.log('layer');
	(async () => {
		await model.fit(trainTensor.sizeMB, trainTensor.timeSec, { epochs: 200 }); //fit == 훈련
		console.log('손실함수');

		model.evaluate(testTensor.sizeMB, testTensor.timeSec).print(); // 손실함수 계산

		const avgDealySec = tf.mean(trainTensor.timeSec);
		console.log('훈련데이터 평균');
		avgDealySec.print(); // 평균을 구해본다

		console.log('테스트 데이터 평균');
		testTensor.timeSec.sub(0.295).abs().mean().print();

		console.log('prediect');
		model.predict(tf.tensor2d([[1], [100]])).print();
	})();
</script>
