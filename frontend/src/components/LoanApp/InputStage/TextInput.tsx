import {Dispatch, FC, SetStateAction} from "react";

type InputBoxProps = {
	type: "text" | "numeric",
	label: string,
	setState: Dispatch<SetStateAction<number>>,
	state: number
}

export const TextInput: FC<InputBoxProps> = ({
	type, label, state, setState
                                            }) => {
	return (
		<div className={"flex flex-col gap-1.5"}>
			<p>{label}:</p>
			<input
				inputMode={type}
				className={"p-2 ml-1 border-gray-400 border shadow-xl rounded-md w-full h-full bg-white"}
				value={state}
				onBeforeInput={(event) => {
					let element = event.target as HTMLInputElement
					if (Number.isNaN(Number(element.value + event.data))) {
						event.preventDefault()
					}
				}}
				onInput={(event)=>{
					let element = event.target as HTMLInputElement
					let val = Number(element.value)
					setState(val)
				}}
			/>
		</div>
	);
}
