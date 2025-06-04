import {Dispatch, SetStateAction, useEffect, useState} from "react";
import {TextInput} from "@/components/LoanApp/InputStage/TextInput";
import {DropdownInput} from "@/components/LoanApp/InputStage/DropdownInput";
import {Checkbox} from "@/components/LoanApp/InputStage/Checkbox";
import {ApiRequestBodySchema, Stage} from "@/components/LoanApp";
import {z} from "zod";
import {Button} from "@/components/Button";


interface InputStageProps {
	setData: Dispatch<SetStateAction<z.infer<typeof ApiRequestBodySchema> | null>>,
	data: z.infer<typeof ApiRequestBodySchema> | null,
	setResult: Dispatch<SetStateAction<string | null>>
	setStage: Dispatch<SetStateAction<Stage>>,
	stage: Stage
}

export enum HomeOwnershipStatus {
	"Own" = "OWN",
	"Rent" = "RENT",
	"Mortgage" = "MORTGAGE"
}

export enum LoanIntent {
	"Education" = "EDUCATION",
	"Venture" = "VENTURE",
	"Personal" = "PERSONAL",
	"Medical" = "MEDICAL",
	"Home Improvement" = "HOMEIMPROVEMENT",
	"Debt Consolidation" = "DEBTCONSOLIDATION"
}

export enum Model {
	"LightGBM" = "lgb",
	"Gradient Boosting" = "gb",
	"Neural Network" = "nn",
	"XGBoost" = "xgb",
	"SVM" = "svm",
	"Random Forest" = "rf"
}

export default function InputStage({
	                                   setData,
	setStage,
	stage,
	data,
	setResult
                                   }: InputStageProps) {


	const [income, setIncome] = useState(data?.params?.income ?? 0)
	const [age, setAge] = useState(data?.params?.age ?? 20)
	const [empLen, setEmpLen] = useState(data?.params?.employment_length ?? 1)
	const [loanAmt, setLoanAmt] = useState(data?.params?.loan_amount ?? 10000)
	const [homeStat, setHomeStat] = useState<HomeOwnershipStatus>(data?.params?.home_ownership as HomeOwnershipStatus ?? HomeOwnershipStatus["Own"])
	const [intent, setIntent] = useState<LoanIntent>(data?.params?.loan_intent as LoanIntent ?? LoanIntent["Education"])
	const [defaultOnFile, setDefaultOnFile] = useState(data?.params?.default_on_file ?? false)
	const [model, setModel] = useState<Model>(data?.model as Model ?? Model["Gradient Boosting"])



	useEffect(()=> {
		setData({
			model,
			params: {
				income,
				age,
				default_on_file: defaultOnFile,
				employment_length: empLen,
				home_ownership: homeStat,
				loan_amount: loanAmt,
				loan_intent: intent
			}
		})
	}, [income, age, empLen, loanAmt, homeStat, intent, defaultOnFile])



	return (
		<div className={"bg-gray-100 rounded-md h-max p-8 flex flex-col justify-between items-center gap-10 text-black"}>
			<div>
				<p className={"text-4xl"}>Credit Score Predictor</p>
			</div>
			<div className={"grid gap-x-6 gap-y-4 grid-cols-2 "}>
				<TextInput type={"numeric"} label={"Income (Yearly)"} state={income} setState={setIncome}/>
				<DropdownInput label={"Loan Intent"} setState={setIntent} options={LoanIntent}/>
				<TextInput type={"numeric"} label={"Age"} state={age} setState={setAge}/>
				<DropdownInput label={"Home Ownership Status"} setState={setHomeStat} options={HomeOwnershipStatus}/>
				<TextInput type={"text"} label={"Employment Length"} state={empLen} setState={setEmpLen}/>
				<Checkbox label={"Defaulted on File"} state={defaultOnFile} setState={setDefaultOnFile}/>
				<TextInput type={"text"} label={"Loan Amount"} state={loanAmt} setState={setLoanAmt}/>
				<DropdownInput label={"Model"} setState={setModel} options={Model}/>
			</div>
			<div className={"w-full"}>
				<Button onClick={async ()=> {
					setStage("processing")
					let response = await fetch(`https://api-credit.umceko.com/predict`, {
						body: JSON.stringify(data),
						method: "POST",
						headers: {
							"Content-Type": "application/json"
						}
					})
					let result = z.object({
						result: z.string()
					}).parse(await response.json())
					setResult(result.result)
					setStage("completed")
				}}
        disabled={stage === "processing"}
        label={"Confirm!"}/>
			</div>
		</div>
	);
}
